import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from easydict import EasyDict as edict 

from .utils import get_betas_by_name, extract


class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.

    Args:
        backbone: Unet backbone.
        image_size: Image size.
        timesteps: Number of diffusion timesteps.
        beta_schedule: Beta schedule name.
    """
    def __init__(
        self,
        backbone,
        timesteps = 1000,
        beta_schedule = 'linear',
    ):
        self.backbone = backbone
        self.timesteps = timesteps
        self.beta_schedule = beta_schedule
        self.backbone_args = edict(inspect.signature(self.backbone.module.forward if hasattr(self.backbone, 'module') else self.backbone.forward).parameters)

        # Diffusion parameters.
        ## Use float64 for accuracy.
        betas = get_betas_by_name(self.beta_schedule, self.timesteps).astype(np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all(), "betas must be in (0, 1]"

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)

    def diffuse(self, x_0, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.
        In other words, sample from q(x_t | x_0).

        Args:
            x_0: The [N x C x ...] tensor of noiseless inputs.
            t: The [N] tensor of diffusion steps (minus 1). Here, 0 means one step.
            noise: If specified, use this noise instead of generating new noise.

        Returns:
            x_t, the noisy version of x_0 under timestep t.
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        assert noise.shape == x_0.shape, "noise must have same shape as x_0"
        return (
            extract(self.sqrt_alphas_cumprod, t, x_0.shape) * x_0
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape) * noise
        )

    def reverse_diffuse(self, x_t, t, noise):
        """
        Get original image from noisy version under timestep t.
        """
        assert noise.shape == x_t.shape, "noise must have same shape as x_t"
        return (
            (x_t - extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * noise)
            / extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        )

    @torch.no_grad()
    def model_inference(self, x, t, classes=None, **kwargs):
        """
        Do inference with the backbone, returning the predicted noise.

        Args:
            x: The [N x C x ...] tensor of noisy inputs at time t.
            t: The [N] tensor of diffusion steps (minus 1). Here, 0 means one step.
            classes: The [N] tensor of class labels.
        
        Returns:
            The [N x C x ...] tensor of predicted noise.
        """
        # filter kwargs
        kwargs = {k: v for k, v in kwargs.items() if k in self.backbone_args}
        return self.backbone(x, t, classes, **kwargs)

    def training_losses(self, x_0, classes=None, **kwargs):
        """
        Compute training losses for a single timestep.

        Args:
            x_0: The [N x C x ...] tensor of noiseless inputs.
            classes: The [N] tensor of class labels.
            kwargs: Additional arguments to pass to the backbone.

        Returns:
            a dict with the key "loss" containing a tensor of shape [N].
            may also contain other keys for different terms.
        """
        noise = torch.randn_like(x_0)
        t = torch.randint(0, self.timesteps, (x_0.shape[0],), device=x_0.device).long()
        x_t = self.diffuse(x_0, t, noise=noise)

        pred_eps = self.backbone(x_t, t, classes, **kwargs)
        assert pred_eps.shape == noise.shape == x_0.shape

        terms = edict()
        terms["mse"] = F.mse_loss(pred_eps, noise)
        terms["loss"] = terms["mse"]
        return terms
