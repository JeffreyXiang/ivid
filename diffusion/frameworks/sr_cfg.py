import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from easydict import EasyDict as edict 

from .gaussian_diffusion import GaussianDiffusion


class SuperResCFG(GaussianDiffusion):
    """
    Image super-resolution with classifier-free guidance.
    
    Args:
        p_uncond: probability of drop the class label.
    """

    def __init__(self, backbone, *, p_uncond=0.1, **kwargs):
        super().__init__(backbone, **kwargs)
        self.p_uncond = p_uncond

    def make_cond_inputs(self, x, y, **kwargs):
        """
        Make inputs for the conditional model.

        Args:
            x: The [N x C x ...] tensor of noisy inputs at time t.
            y: The [N x C x ...] tensor of low-resolution images.
        """
        
        scale = x.shape[-1] // y.shape[-1]
        y = F.interpolate(y, scale_factor=scale, mode='bilinear', align_corners=False)
        in_list = [x, y]

        return torch.cat(in_list, dim=1)


    @torch.no_grad()
    def model_inference(self, x, t, y, classes=None, strength=3.0, **kwargs):
        """
        Do inference with the backbone, returning the predicted noise.

        Args:
            x: The [N x C x ...] tensor of noisy inputs at time t.
            t: The [N] tensor of diffusion steps (minus 1). Here, 0 means one step.
            y: The [N x C x ...] tensor of ground truth partially visible images.
            classes: The [N] tensor of class labels.
            strength: The strength of the classifier-free guidance.
        
        Returns:
            The [N x C x ...] tensor of predicted noise.
        """
        cond_inputs = self.make_cond_inputs(x, y, **kwargs)
        if classes is None:
            return self.backbone(cond_inputs, t, None)
        return (
            (1 + strength) * self.backbone(cond_inputs, t, classes)
            - (strength * self.backbone(cond_inputs, t, None) if strength > 0 else 0)
        )

    def training_losses(self, x_0, y, classes=None, **kwargs):
        """
        Compute training losses for a single timestep.

        Args:
            x_0: The [N x C x ...] tensor of noiseless inputs.
            y: The [N x C x ...] tensor of low-resolution images.
            classes: The [N] tensor of class labels.

        Returns:
            a dict with the key "loss" containing a tensor of shape [N].
            may also contain other keys for different terms.
        """
        noise = torch.randn_like(x_0)
        t = torch.randint(0, self.timesteps, (x_0.shape[0],), device=x_0.device).long()
        x_t = self.diffuse(x_0, t, noise=noise)

        # randomly drop the class label
        if classes is not None and self.p_uncond > 0:
            classes = torch.where(
                torch.rand_like(classes.float()) < self.p_uncond,
                -torch.ones_like(classes),
                classes,
            )

        x_t = self.make_cond_inputs(x_t, y, **kwargs)

        pred_eps = self.backbone(x_t, t, classes)
        assert pred_eps.shape == noise.shape == x_0.shape

        terms = edict()
        terms["mse"] = F.mse_loss(pred_eps, noise)
        terms["loss"] = 1 * terms["mse"]

        return terms
