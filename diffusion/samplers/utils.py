import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def extract(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    Args:
        arr: the 1-D numpy array.
        timesteps: a tensor of indices into the array to extract.
        broadcast_shape: a larger shape of K dimensions with the batch
            dimension equal to the length of timesteps.

    Returns:
        A tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


def gaussian_blur(x, sigma=0.6, kernel_size=3):
    ch = x.shape[1]
    kernel = torch.exp(-torch.arange(-(kernel_size // 2), kernel_size // 2 + 1) ** 2 / (2 * sigma ** 2)).type(torch.float32).to(x.device)
    kernel = kernel / kernel.sum()
    kernel = kernel[:, None] @ kernel[None, :]
    kernel = kernel[None, None, ...].repeat(ch, 1, 1, 1)
    padding = kernel_size // 2
    x = F.pad(x, (padding, padding, padding, padding), mode='replicate')
    x = F.conv2d(x, kernel, groups=ch)
    return x
