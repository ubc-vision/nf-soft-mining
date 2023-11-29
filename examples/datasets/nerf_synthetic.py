"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

import collections
import json
import os

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F

from .utils import Rays, sample_from_pdf_with_indices, \
                   compute_sobel_edge, Image


def _load_renderings(root_fp: str, subject_id: str, split: str):
    """Load images from disk."""
    if not root_fp.startswith("/"):
        # allow relative path. e.g., "./data/nerf_synthetic/"
        root_fp = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "..",
            root_fp,
        )

    data_dir = os.path.join(root_fp, subject_id)
    with open(
        os.path.join(data_dir, "transforms_{}.json".format(split)), "r"
    ) as fp:
        meta = json.load(fp)
    images = []
    camtoworlds = []

    for i in range(len(meta["frames"])):
        frame = meta["frames"][i]
        fname = os.path.join(data_dir, frame["file_path"] + ".png")
        rgba = imageio.imread(fname)
        camtoworlds.append(frame["transform_matrix"])
        images.append(rgba)

    images = np.stack(images, axis=0)
    camtoworlds = np.stack(camtoworlds, axis=0)

    h, w = images.shape[1:3]
    camera_angle_x = float(meta["camera_angle_x"])
    focal = 0.5 * w / np.tan(0.5 * camera_angle_x)

    return images, camtoworlds, focal


class SubjectLoader(torch.utils.data.Dataset):
    """Single subject data loader for training and evaluation."""

    SPLITS = ["train", "val", "trainval", "test"]
    SUBJECT_IDS = [
        "chair",
        "drums",
        "ficus",
        "hotdog",
        "lego",
        "materials",
        "mic",
        "ship",
    ]

    WIDTH, HEIGHT = 800, 800
    NEAR, FAR = 2.0, 6.0
    OPENGL_CAMERA = True

    def __init__(
        self,
        subject_id: str,
        root_fp: str,
        split: str,
        color_bkgd_aug: str = "white",
        num_rays: int = None,
        near: float = None,
        far: float = None,
        batch_over_images: bool = True,
        device: torch.device = torch.device("cpu"),
        sampling_type: str = "uniform",
        minpct: float = 0.1,
        lossminpc: float = 0.1,
    ):
        super().__init__()
        assert split in self.SPLITS, "%s" % split
        assert subject_id in self.SUBJECT_IDS, "%s" % subject_id
        assert color_bkgd_aug in ["white", "black", "random"]
        self.split = split
        self.num_rays = num_rays
        self.near = self.NEAR if near is None else near
        self.far = self.FAR if far is None else far
        self.training = (num_rays is not None) and (
            split in ["train", "trainval"]
        )
        self.color_bkgd_aug = color_bkgd_aug
        self.batch_over_images = batch_over_images
        if split == "trainval":
            _images_train, _camtoworlds_train, _focal_train = _load_renderings(
                root_fp, subject_id, "train"
            )
            _images_val, _camtoworlds_val, _focal_val = _load_renderings(
                root_fp, subject_id, "val"
            )
            self.images = np.concatenate([_images_train, _images_val])
            self.camtoworlds = np.concatenate(
                [_camtoworlds_train, _camtoworlds_val]
            )
            self.focal = _focal_train
        else:
            self.images, self.camtoworlds, self.focal = _load_renderings(
                root_fp, subject_id, split
            )
        self.images = torch.from_numpy(self.images).to(torch.uint8)
        self.camtoworlds = torch.from_numpy(self.camtoworlds).to(torch.float32)
        self.K = torch.tensor(
            [
                [self.focal, 0, self.WIDTH / 2.0],
                [0, self.focal, self.HEIGHT / 2.0],
                [0, 0, 1],
            ],
            dtype=torch.float32,
        )  # (3, 3)
        self.images = self.images.to(device)
        self.camtoworlds = self.camtoworlds.to(device)
        self.K = self.K.to(device)
        assert self.images.shape[1:3] == (self.HEIGHT, self.WIDTH)
        self.sampling_type = sampling_type
        self.simages = Image(self.images, device=device)
        if self.training:
            bs = np.ceil(self.num_rays / self.images.shape[0]).astype(np.int32) 
            self.num_rays = bs * self.images.shape[0]
            self.const_img_id = torch.arange(0, self.images.shape[0], device=device).repeat_interleave(bs)

        if self.sampling_type == "lmc":
            self.image_edges = compute_sobel_edge(self.images.float()).reshape(self.images.shape[0], -1)
            self.image_edges = self.image_edges.to(device)
            probs = self.image_edges / self.image_edges.sum(dim=-1, keepdim=True)
            cdf = torch.cumsum(probs, dim=-1)
            cdf = torch.nn.functional.pad(cdf, pad=(1, 0), mode='constant', value=0)
            self.cdf = cdf.view(cdf.shape[0], -1)

            self.rand_ten = torch.empty((self.num_rays, 2), dtype=torch.float32, device=device)
            self.noise = torch.empty((self.num_rays, 2), dtype=torch.float32, device=device)

            self.u_num = int(minpct * self.num_rays)
            self.reinit = int(lossminpc * self.num_rays)
            self.prev_samples = None
            x = torch.randint(0, self.WIDTH, size=(self.num_rays,), device=device)
            y = torch.randint(0, self.HEIGHT, size=(self.num_rays,), device=device)
            x = x.float() / (self.WIDTH - 1)
            y = y.float() / (self.HEIGHT - 1)
            self.prev_samples = torch.cat([y[..., None], x[..., None]], dim=1)
            self.prev_samples.clamp_(min=0.0, max=1.0)
            self.HW = torch.tensor([self.HEIGHT-1, self.WIDTH-1], device=device)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index, net_grad=None, loss_per_pix=None):
        if self.sampling_type == "uniform":
            data = self.fetch_data(index)
        elif self.sampling_type == "lmc":
            data = self.fetch_data_lmc(net_grad=net_grad, loss_per_pix=loss_per_pix)
        data = self.preprocess(data)
        return data

    def preprocess(self, data):
        """Process the fetched / cached data with randomness."""
        rgba, rays = data["rgba"], data["rays"]
        pixels, alpha = torch.split(rgba, [3, 1], dim=-1)

        if self.training:
            if self.color_bkgd_aug == "random":
                color_bkgd = torch.rand(3, device=self.images.device)
            elif self.color_bkgd_aug == "white":
                color_bkgd = torch.ones(3, device=self.images.device)
            elif self.color_bkgd_aug == "black":
                color_bkgd = torch.zeros(3, device=self.images.device)
        else:
            # just use white during inference
            color_bkgd = torch.ones(3, device=self.images.device)

        pixels = pixels * alpha + color_bkgd * (1.0 - alpha)
        return {
            "pixels": pixels,  # [n_rays, 3] or [h, w, 3]
            "rays": rays,  # [n_rays,] or [h, w]
            "color_bkgd": color_bkgd,  # [3,]
            **{k: v for k, v in data.items() if k not in ["rgba", "rays"]},
        }

    def update_num_rays(self, num_rays):
        self.num_rays = num_rays

    def get_origin_viewdirs(self, image_id, y, x):
        c2w = self.camtoworlds[image_id]  # (num_rays, 3, 4)
        camera_dirs = F.pad(
            torch.stack(
                [
                    (x - self.K[0, 2] + 0.5) / self.K[0, 0],
                    (y - self.K[1, 2] + 0.5)
                    / self.K[1, 1]
                    * (-1.0 if self.OPENGL_CAMERA else 1.0),
                ],
                dim=-1,
            ),
            (0, 1),
            value=(-1.0 if self.OPENGL_CAMERA else 1.0),
        )  # [num_rays, 3]

        # [n_cams, height, width, 3]
        directions = (camera_dirs[:, None, :] * c2w[:, :3, :3]).sum(dim=-1)
        origins = torch.broadcast_to(c2w[:, :3, -1], directions.shape)
        viewdirs = directions / torch.linalg.norm(
            directions, dim=-1, keepdims=True
        )
        return origins, viewdirs

    def fetch_data(self, index):
        """Fetch the data (it maybe cached for multiple batches)."""
        num_rays = self.num_rays

        if self.training:
            if self.batch_over_images:
                image_id = torch.randint(
                    0,
                    len(self.images),
                    size=(num_rays,),
                    device=self.images.device,
                )
            else:
                image_id = [index] * num_rays
            x = torch.randint(
                0, self.WIDTH, size=(num_rays,), device=self.images.device
            )
            y = torch.randint(
                0, self.HEIGHT, size=(num_rays,), device=self.images.device
            )
        else:
            image_id = [index]
            x, y = torch.meshgrid(
                torch.arange(self.WIDTH, device=self.images.device),
                torch.arange(self.HEIGHT, device=self.images.device),
                indexing="xy",
            )
            x = x.flatten()
            y = y.flatten()

        # generate rays
        rgba = self.images[image_id, y, x] / 255.0  # (num_rays, 4)
        origins, viewdirs = self.get_origin_viewdirs(image_id, y, x)

        if self.training:
            origins = torch.reshape(origins, (num_rays, 3))
            viewdirs = torch.reshape(viewdirs, (num_rays, 3))
            rgba = torch.reshape(rgba, (num_rays, 4))
        else:
            origins = torch.reshape(origins, (self.HEIGHT, self.WIDTH, 3))
            viewdirs = torch.reshape(viewdirs, (self.HEIGHT, self.WIDTH, 3))
            rgba = torch.reshape(rgba, (self.HEIGHT, self.WIDTH, 4))

        rays = Rays(origins=origins, viewdirs=viewdirs)

        return {
            "rgba": rgba,  # [h, w, 4] or [num_rays, 4]
            "rays": rays,  # [h, w, 3] or [num_rays, 3]
        }

    def fetch_data_lmc(self, net_grad=None, a=2e1, b=2e-2, loss_per_pix=None):
        if net_grad is not None:
            with torch.no_grad():
                self.noise.normal_(mean=0.0, std=1.0)
                self.rand_ten.uniform_()

                net_grad.mul_(a).add_(self.noise, alpha=b)
                self.prev_samples.add_(net_grad)

                threshold, _ = torch.topk(loss_per_pix, self.reinit+1, largest=False)
                mask = loss_per_pix <= threshold[-1]
                mask = torch.cat([self.prev_samples < 0, 
                                  self.prev_samples > 1,
                                  loss_per_pix.unsqueeze(1) <= threshold[-1]
                                 ], 1)
                mask = mask.sum(1)
                bound_idxs = torch.where(mask)[0]
                self.prev_samples[-self.u_num:].copy_(self.rand_ten[-self.u_num:])
                
                if bound_idxs.shape[0] > 0:
                    # sample from edges
                    count = torch.bincount(self.const_img_id[bound_idxs], minlength=self.images.shape[0])
                    batch1d = sample_from_pdf_with_indices(self.cdf, int(self.num_rays / self.images.shape[0]))
                    indices = torch.arange(batch1d.size(1), device=batch1d.device).unsqueeze_(0).repeat(batch1d.size(0), 1)
                    mask = indices < count.unsqueeze(1)        
                    batch1d = batch1d.masked_select(mask)

                    self.prev_samples[bound_idxs, 0] = (batch1d // self.WIDTH) / (self.HEIGHT - 1) 
                    self.prev_samples[bound_idxs, 1] = (batch1d % self.WIDTH) / (self.WIDTH - 1)
                self.prev_samples.clamp_(min=0.0, max=1.0)
        points_2d = self.prev_samples *  self.HW
        points_2d.round_()     

        # generate rays
        rgba = self.simages(self.const_img_id, points_2d[:, 0], points_2d[:, 1])  / 255.0
        points_2d.requires_grad = True
        x, y = points_2d[:, 1], points_2d[:, 0]
        origins, viewdirs = self.get_origin_viewdirs(self.const_img_id, y, x)

        if self.training:
            origins = torch.reshape(origins, (self.num_rays, 3))
            viewdirs = torch.reshape(viewdirs, (self.num_rays, 3))
            rgba = torch.reshape(rgba, (self.num_rays, 4))
        else:
            origins = torch.reshape(origins, (self.HEIGHT, self.WIDTH, 3))
            viewdirs = torch.reshape(viewdirs, (self.HEIGHT, self.WIDTH, 3))
            rgba = torch.reshape(rgba, (self.HEIGHT, self.WIDTH, 4))

        rays = Rays(origins=origins, viewdirs=viewdirs)

        return {
            "rgba": rgba,  # [h, w, 4] or [num_rays, 4]
            "rays": rays,  # [h, w, 3] or [num_rays, 3]
            "x": x,
            "y": y,
            "image_id": self.const_img_id,
            "points_2d": points_2d
        }
