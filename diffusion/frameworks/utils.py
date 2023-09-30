import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def get_betas_by_name(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.
    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.

    Args:
        schedule_name (str): Name of the beta schedule.
        num_diffusion_timesteps (int): Number of diffusion timesteps.

    Returns:
        betas (np.ndarray): A numpy array of betas.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: np.cos((t + 0.008) / 1.008 * np.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    Args:
        num_diffusion_timesteps: the number of betas to produce.
        alpha_bar: a lambda that takes an argument t from 0 to 1 and
            produces the cumulative product of (1-beta) up to that
            part of the diffusion process.
        max_beta: the maximum beta to use; use values lower than 1 to
            prevent singularities.

    Returns:
        A numpy array of betas.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


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
