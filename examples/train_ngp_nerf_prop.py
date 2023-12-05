"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""
import argparse
import itertools
import pathlib
import time
from typing import Callable

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from lpips import LPIPS
from radiance_fields.ngp import NGPDensityField, NGPRadianceField
from losses import NeRFLoss
from schedulers import create_scheduler
torch.autograd.set_detect_anomaly(True)

from examples.utils import (
    LLFF_NDC_SCENES,
    NERF_SYNTHETIC_SCENES,
    render_image_with_propnet,
    set_random_seed,
)
from nerfacc.estimators.prop_net import (
    PropNetEstimator,
    get_proposal_requires_grad_fn,
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_root",
    type=str,
    default=str("/ubc/cs/research/kmyi/shakiba/g/data/nerf_llff_data"),
    # default=str("../../data/nerf_synthetic"),
    help="the root dir of the dataset",
)
parser.add_argument(
    "--train_split",
    type=str,
    default="train",
    choices=["train", "trainval"],
    help="which train split to use",
)
parser.add_argument(
    "--scene",
    type=str,
    default="trex",
    choices=NERF_SYNTHETIC_SCENES + LLFF_NDC_SCENES,
    help="which scene to use",
)
parser.add_argument(
    "--test_chunk_size",
    type=int,
    default=8192,
)
parser.add_argument(
    "--sampling_type",
    type=str,
    choices=["uniform", "lmc"],
    default="lmc",
)
parser.add_argument(
    "--i_print",
    type=int,
    default=1000,
)
parser.add_argument(
    "--scheduler",
    type=str,
    default="cosineannealing",
)
parser.add_argument(
    "--minpct",
    type=float,
    default=0.1,
)
parser.add_argument(
    "--lossminpc",
    type=float,
    default=0.1,
)
args = parser.parse_args()

device = "cuda:0"
set_random_seed(42)

if args.scene in LLFF_NDC_SCENES:
    from datasets.nerf_llff import SubjectLoader
     # training parameters
    max_steps = 20000
    init_batch_size = 4096
    unbounded = 2
    weight_decay = 1e-5 
    # scene parameters
    aabb = torch.tensor([-1.5, -1.5, -1.5, 1.5, 1.5, 1.5], device=device)
    near_plane = 0
    far_plane = 1
    # dataset parameters
    train_dataset_kwargs = {"sampling_type": args.sampling_type, 
                            "minpct": args.minpct, "lossminpc": args.lossminpc}
    test_dataset_kwargs = {}
    # model parameters
    proposal_networks = [
        NGPDensityField(
            aabb=aabb,
            unbounded=unbounded,
            n_levels=5,
            max_resolution=128,
        ).to(device),
    ]
    # render parameters
    num_samples = 64
    num_samples_per_prop = [128]
    prop_sampling_type = "uniform"
    opaque_bkgd = False

    # lmc
    alpha = 0.8
else:
    from datasets.nerf_synthetic import SubjectLoader

    # training parameters
    max_steps = 20000
    init_batch_size = 4096
    weight_decay = (
        1e-5 if args.scene in ["materials", "ficus", "drums"] else 1e-6
    )
    # scene parameters
    unbounded = False
    aabb = torch.tensor([-1.5, -1.5, -1.5, 1.5, 1.5, 1.5], device=device)
    near_plane = 2.0
    far_plane = 6.0
    # dataset parameters
    train_dataset_kwargs = {"sampling_type": args.sampling_type, "minpct": args.minpct, "lossminpc": args.lossminpc}
    test_dataset_kwargs = {}
    # model parameters
    proposal_networks = [
        NGPDensityField(
            aabb=aabb,
            unbounded=unbounded,
            n_levels=5,
            max_resolution=128,
        ).to(device),
    ]
    # render parameters
    num_samples = 64
    num_samples_per_prop = [128]
    prop_sampling_type = "uniform"
    opaque_bkgd = False

    # lmc
    alpha = 0.6

train_dataset = SubjectLoader(
    subject_id=args.scene,
    root_fp=args.data_root,
    split=args.train_split,
    num_rays=init_batch_size,
    device=device,
    **train_dataset_kwargs,
)

test_dataset = SubjectLoader(
    subject_id=args.scene,
    root_fp=args.data_root,
    split="test",
    num_rays=None,
    device=device,
    **test_dataset_kwargs,
)

# setup the radiance field we want to train.
prop_optimizer = torch.optim.Adam(
    itertools.chain(
        *[p.parameters() for p in proposal_networks],
    ),
    lr=1e-2,
    eps=1e-15,
    weight_decay=weight_decay,
)
prop_scheduler = create_scheduler(prop_optimizer, args.scheduler, max_steps, 1e-2)
estimator = PropNetEstimator(prop_optimizer, prop_scheduler).to(device)

grad_scaler = torch.cuda.amp.GradScaler(2**10)
radiance_field = NGPRadianceField(aabb=aabb, unbounded=unbounded).to(device)
optimizer = torch.optim.Adam(
    radiance_field.parameters(),
    lr=1e-2,
    eps=1e-15,
    weight_decay=weight_decay,
)
scheduler = create_scheduler(optimizer, args.scheduler, max_steps, 1e-2)
proposal_requires_grad_fn = get_proposal_requires_grad_fn()
# proposal_annealing_fn = get_proposal_annealing_fn()

lpips_net = LPIPS(net="vgg").to(device)
lpips_norm_fn = lambda x: x[None, ...].permute(0, 3, 1, 2) * 2 - 1
lpips_fn = lambda x, y: lpips_net(lpips_norm_fn(x), lpips_norm_fn(y)).mean()

loss_fn = NeRFLoss(lambda_distortion=1e-1, lambda_opacity=1e-3)

gradval = None
lossperpix_prev = None

# training
tic = time.time()
for step in range(max_steps + 1):
    radiance_field.train()
    for p in proposal_networks:
        p.train()
    estimator.train()

    i = torch.randint(0, len(train_dataset), (1,)).item()
    data = train_dataset.__getitem__(i, net_grad=gradval, loss_per_pix=lossperpix_prev)


    render_bkgd = data["color_bkgd"]
    rays = data["rays"]
    pixels = data["pixels"]

    proposal_requires_grad = proposal_requires_grad_fn(step)
    # render
    rgb, acc, depth, extras, distkwargs = render_image_with_propnet(
        radiance_field,
        proposal_networks,
        estimator,
        rays,
        # rendering options
        num_samples=num_samples,
        num_samples_per_prop=num_samples_per_prop,
        near_plane=near_plane,
        far_plane=far_plane,
        sampling_type=prop_sampling_type,
        opaque_bkgd=opaque_bkgd,
        render_bkgd=render_bkgd,
        # train options
        proposal_requires_grad=proposal_requires_grad,
    )
    estimator.update_every_n_steps(
        extras["trans"], proposal_requires_grad, loss_scaler=1024
    )

    # compute loss
    loss_d = loss_fn(rgb, pixels, acc, distkwargs)
    loss_per_pix = loss_d['rgb'].mean(-1)
    if 'opacity' in loss_d:
        loss_per_pix = loss_per_pix + loss_d['opacity'].squeeze(-1)
    if 'distorion' in loss_d:
        loss_per_pix = loss_per_pix + loss_d['distortion']

    if args.sampling_type in ["lmc"]:
        imp_loss = torch.abs(rgb - pixels).mean(-1).detach()
        if 'opacity' in loss_d:
            imp_loss = (imp_loss + loss_d['opacity'].squeeze(-1)).detach()
        if 'distorion' in loss_d:
            imp_loss = imp_loss + loss_d['distortion'].detach()
        correction = 1.0 / torch.clip(imp_loss, min=torch.finfo(torch.float16).eps).detach()
        if alpha != 0:
            r = min((step/1000), alpha)
        else:
            r = alpha
        correction.pow_(r)
        correction.clamp_(min=0.2, max=correction.mean()+correction.std())
        loss_per_pix.mul_(correction) 
    loss = loss_per_pix.mean()

    optimizer.zero_grad()
    # do not unscale it because we are using Adam.
    grad_scaler.scale(loss).backward()
    optimizer.step()
    scheduler.step()

    if args.sampling_type == "lmc":
        with torch.no_grad():            
            net_grad = data['points_2d'].grad.detach()
            loss_per_pix = loss_per_pix.detach()
            net_grad = net_grad / ((grad_scaler._scale * (correction * loss_per_pix).unsqueeze(1))+ torch.finfo(net_grad.dtype).eps)
            gradval = net_grad
            lossperpix_prev = loss_per_pix

    if step % args.i_print == 0:
        elapsed_time = time.time() - tic
        loss = F.mse_loss(rgb, pixels)
        psnr = -10.0 * torch.log(loss) / np.log(10.0)
        print(
            f"elapsed_time={elapsed_time:.2f}s | step={step} | "
            f"loss={loss:.5f} | psnr={psnr:.2f} | "
            f"num_rays={len(pixels):d} | "
            f"max_depth={depth.max():.3f} | "
        )

    if step > 0 and (step % args.i_print == 0 or step % max_steps == 0):
        # evaluation
        radiance_field.eval()
        for p in proposal_networks:
            p.eval()
        estimator.eval()

        psnrs = []
        lpips = []
        with torch.no_grad():
            for i in tqdm.tqdm(range(len(test_dataset))):
                data = test_dataset[i]
                render_bkgd = data["color_bkgd"]
                rays = data["rays"]
                pixels = data["pixels"]

                # rendering
                rgb, acc, depth, _, _ = render_image_with_propnet(
                    radiance_field,
                    proposal_networks,
                    estimator,
                    rays,
                    # rendering options
                    num_samples=num_samples,
                    num_samples_per_prop=num_samples_per_prop,
                    near_plane=near_plane,
                    far_plane=far_plane,
                    sampling_type=prop_sampling_type,
                    opaque_bkgd=opaque_bkgd,
                    render_bkgd=render_bkgd,
                    # test options
                    test_chunk_size=args.test_chunk_size,
                )
                mse = F.mse_loss(rgb, pixels)
                psnr = -10.0 * torch.log(mse) / np.log(10.0)
                psnrs.append(psnr.item())
                lpips.append(lpips_fn(rgb, pixels).item())
                # if i == 0:
                #     imageio.imwrite(
                #         "rgb_test.png",
                #         (rgb.cpu().numpy() * 255).astype(np.uint8),
                #     )
                #     imageio.imwrite(
                #         "rgb_error.png",
                #         (
                #             (rgb - pixels).norm(dim=-1).cpu().numpy() * 255
                #         ).astype(np.uint8),
                #     )
                #     break
        psnr_avg = sum(psnrs) / len(psnrs)
        lpips_avg = sum(lpips) / len(lpips)
        print(f"evaluation: psnr_avg={psnr_avg}, lpips_avg={lpips_avg}")
