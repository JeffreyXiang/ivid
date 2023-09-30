import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from easydict import EasyDict as edict 

from .gaussian_diffusion import GaussianDiffusion


class ClassifierFreeGuidance(GaussianDiffusion):
    """
    Diffusion model with classifier-free guidance.
    
    Args:
        p_uncond: probability of drop the class label.
    """
    def __init__(self, backbone, *, p_uncond=0.1, **kwargs):
        super().__init__(backbone, **kwargs)
        self.p_uncond = p_uncond

    @torch.no_grad()
    def model_inference(self, x, t, classes=None, strength=3.0, **kwargs):
        """
        Do inference with the backbone, returning the predicted noise.

        Args:
            x: The [N x C x ...] tensor of noisy inputs at time t.
            t: The [N] tensor of diffusion steps (minus 1). Here, 0 means one step.
            classes: The [N] tensor of class labels.
            strength: The strength of the classifier-free guidance.
        
        Returns:
            The [N x C x ...] tensor of predicted noise.
        """
        # filter kwargs
        kwargs = {k: v for k, v in kwargs.items() if k in self.backbone_args}
        return (
            (1 + strength) * self.backbone(x, t, classes, **kwargs)
            - (strength * self.backbone(x, t, None, **kwargs) if strength > 0 else 0)
        )

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

        # randomly drop the class label
        if self.p_uncond > 0:
            classes = torch.where(
                torch.rand_like(classes.float()) < self.p_uncond,
                -torch.ones_like(classes),
                classes,
            )

        pred_eps = self.backbone(x_t, t, classes, **kwargs)
        assert pred_eps.shape == noise.shape == x_0.shape

        terms = edict()
        terms["mse"] = F.mse_loss(pred_eps, noise)
        terms["loss"] = terms["mse"]
        return terms
