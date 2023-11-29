"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

import collections
import torch
import torch.nn.functional as F

Rays = collections.namedtuple("Rays", ("origins", "viewdirs"))


def namedtuple_map(fn, tup):
    """Apply `fn` to each element of `tup` and cast to `tup`'s namedtuple."""
    return type(tup)(*(None if x is None else fn(x) for x in tup))


@torch.cuda.amp.autocast(dtype=torch.float64)
def sample_from_pdf_with_indices(cdf, num_points):
    # Normalize the PDFs
    u = torch.rand((cdf.shape[0], num_points,), device=cdf.device, dtype=torch.float64) #* cdf.max()
    batch1d = torch.searchsorted(cdf, u, right=True) - 1
    return batch1d


def compute_sobel_edge(images):
    # Ensure the input is a torch tensor
    if not isinstance(images, torch.Tensor):
        images = torch.tensor(images)
    
    if images.max() > 1.0:
        images = images / 255.0

    # Convert the images to grayscale using weighted sum of channels (shape: N x H x W x 1)
    gray_images = 0.2989 * images[..., 0] + 0.5870 * images[..., 1] + 0.1140 * images[..., 2]
    gray_images = gray_images.unsqueeze(-1)

    # Transpose the images to the shape (N, C, H, W)
    gray_images = gray_images.permute(0, 3, 1, 2)

    # Define Sobel kernels
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=images.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=images.device).view(1, 1, 3, 3)

    # Compute Sobel edges
    edge_x = F.conv2d(gray_images, sobel_x, padding=1)
    edge_y = F.conv2d(gray_images, sobel_y, padding=1)
    edges = torch.sqrt(edge_x ** 2 + edge_y ** 2)

    maxval, _ = edges.max(dim=1)[0].max(dim=1)
    edges = edges / (maxval.unsqueeze(1).unsqueeze(1) + 1e-7)
    edges = torch.clip(edges, min=1e-5, max=1.0)
    return edges.squeeze(1)


# source: https://github.com/NVlabs/tiny-cuda-nn/blob/master/samples/mlp_learning_an_image_pytorch.py
class Image(torch.nn.Module):
    def __init__(self, images, device):
        super(Image, self).__init__()
        self.data = images.to(device, non_blocking=True)
        self.shape = self.data[0].shape

    @torch.cuda.amp.autocast(dtype=torch.float32)
    def forward(self, iind, ys, xs):
        shape = self.shape

        xy = torch.cat([ys.unsqueeze(1), xs.unsqueeze(1)], dim=1)
        indices = xy.long()
        lerp_weights = xy - indices.float() 

        y0 = indices[:, 0].clamp(min=0, max=shape[0]-1)
        x0 = indices[:, 1].clamp(min=0, max=shape[1]-1)
        y1 = (y0 + 1).clamp(max=shape[0]-1)
        x1 = (x0 + 1).clamp(max=shape[1]-1)

        return (
            self.data[iind, y0, x0] * (1.0 - lerp_weights[:,0:1]) * (1.0 - lerp_weights[:,1:2]) +
            self.data[iind, y0, x1] * lerp_weights[:,0:1] * (1.0 - lerp_weights[:,1:2]) +
            self.data[iind, y1, x0] * (1.0 - lerp_weights[:,0:1]) * lerp_weights[:,1:2] +
            self.data[iind, y1, x1] * lerp_weights[:,0:1] * lerp_weights[:,1:2]
        )
