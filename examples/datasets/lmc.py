import torch
from .utils import sample_from_pdf_with_indices, compute_sobel_edge

class LMC(torch.nn.Module):
    def __init__(self, images, num_rays, const_img_id, device, minpct=0.1, lossminpc=0.1, a=2e1, b=2e-2):
        super().__init__()
        self.num_rays = num_rays
        self.a = a
        self.b = b
        self.WIDTH = images.shape[2]
        self.HEIGHT = images.shape[1]
        self.NUM_IMGS = images.shape[0]
        self.const_img_id = const_img_id
        self.image_edges = compute_sobel_edge(images.float()).reshape(images.shape[0], -1)
        self.image_edges = self.image_edges.to(device)
        probs = self.image_edges / self.image_edges.sum(dim=-1, keepdim=True)
        cdf = torch.cumsum(probs, dim=-1)
        cdf = torch.nn.functional.pad(cdf, pad=(1, 0), mode='constant', value=0)
        self.cdf = cdf.view(cdf.shape[0], -1)

        self.rand_ten = torch.empty((self.num_rays, 2), dtype=torch.float32, device=device)
        self.noise = torch.empty((self.num_rays, 2), dtype=torch.float32, device=device)

        self.u_num = int(minpct * self.num_rays)
        self.reinit = int(lossminpc * self.num_rays)
        x = torch.randint(0, self.WIDTH, size=(self.num_rays,), device=device)
        y = torch.randint(0, self.HEIGHT, size=(self.num_rays,), device=device)
        x = x.float() / (self.WIDTH - 1)
        y = y.float() / (self.HEIGHT - 1)
        self.prev_samples = torch.cat([y[..., None], x[..., None]], dim=1)
        self.prev_samples.clamp_(min=0.0, max=1.0)
        self.HW = torch.tensor([self.HEIGHT-1, self.WIDTH-1], device=device)

    def forward(self, net_grad, loss_per_pix):
        if net_grad is not None:
            with torch.no_grad():
                self.noise.normal_(mean=0.0, std=1.0)
                self.rand_ten.uniform_()

                net_grad.mul_(self.a).add_(self.noise, alpha=self.b)
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
                    count = torch.bincount(self.const_img_id[bound_idxs], minlength=self.NUM_IMGS)
                    batch1d = sample_from_pdf_with_indices(self.cdf, int(self.num_rays / self.NUM_IMGS))
                    indices = torch.arange(batch1d.size(1), device=batch1d.device).unsqueeze_(0).repeat(batch1d.size(0), 1)
                    mask = indices < count.unsqueeze(1)        
                    batch1d = batch1d.masked_select(mask)

                    self.prev_samples[bound_idxs, 0] = (batch1d // self.WIDTH) / (self.HEIGHT - 1) 
                    self.prev_samples[bound_idxs, 1] = (batch1d % self.WIDTH) / (self.WIDTH - 1)
                self.prev_samples.clamp_(min=0.0, max=1.0)
        point2d = self.prev_samples *  self.HW
        return point2d.round_()