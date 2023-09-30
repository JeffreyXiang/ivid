import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm
from easydict import EasyDict as edict 

from .utils import extract


class DdimSampler:
    """
    Generate samples from a diffusion model using DDIM

    Args:
        framwork: Diffusion framework.
    """
    def __init__(
        self,
        framework,
    ):
        self.framework = framework

        # Diffusion parameters.
        betas = self.framework.betas
        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_eps_from_xstart(self, x_t, t, x_0):
        assert x_t.shape == x_0.shape
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x_0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    @torch.no_grad()
    def sample_once(
        self,
        x_t,
        t,
        t_prev,
        classes=None,
        clip_denoised=False,
        eta=0.0,
        replace_rgb=None,
        replace_depth=None,
        constrain_depth=None,
        **kwargs
    ):
        """
        Sample x_{t-1} from the model using DDIM.
        
       Args:
            x_t: The [N x C x ...] tensor of noisy inputs at time t.
            t: The [N] tensor of diffusion steps. Note that this is the actual diffusion step. 1 means one step.
            t_prev: The [N] tensor of diffusion steps for the previous timestep.
            classes: The [N] tensor of class labels.
            clip_denoised: If True, clip the denoised signal into [-1, 1].
            eta (float): The DDIM eta parameter.
            replace_rgb (float): Weight of replace the predicted RGB of unmasked pixels with the actual RGB.
            replace_depth (float): Weight of replace the predicted depth of unmasked pixels with the actual depth.
            constrain_depth (float): Weight of constrain the predicted depth of masked pixels to be larger than the convex hull.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'pred_x_prev': x_{t-1}.
            - 'pred_x_0': a prediction of x_0.
        """
        pred_eps = self.framework.model_inference(x_t, t - 1, classes=classes, **kwargs)
        pred_x_0 = self._predict_xstart_from_eps(x_t=x_t, t=t - 1, eps=pred_eps)
        nonzero_mask = (t_prev != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
        if clip_denoised:
            pred_x_0 = torch.clamp(pred_x_0, -1.0, 1.0)
        if replace_rgb is not None:
            replace_weight, replace_rgb, replace_mask = replace_rgb
            pred_x_0[:, :3] = (1 - nonzero_mask) * pred_x_0[:, :3] \
                + nonzero_mask * ((replace_weight * replace_rgb + (1 - replace_weight) * pred_x_0[:, :3]) * replace_mask + pred_x_0[:, :3] * (1 - replace_mask))
        if replace_depth: 
            replace_weight, replace_depth, replace_mask = replace_depth
            pred_x_0[:, 3:] = (replace_weight * replace_depth + (1 - replace_weight) * pred_x_0[:, 3:]) * replace_mask + pred_x_0[:, 3:] * (1 - replace_mask)
            if constrain_depth:
                constrain_weight, convex = constrain_depth
                pred_x_0[:, 3:] = pred_x_0[:, 3:] * replace_mask + (constrain_weight * torch.maximum(pred_x_0[:, 3:], convex) + (1 - constrain_weight) * pred_x_0[:, 3:]) * (1 - replace_mask) 
        pred_eps = self._predict_eps_from_xstart(x_t=x_t, t=t - 1, x_0=pred_x_0)
        alpha_bar = extract(self.alphas_cumprod, t - 1, x_t.shape)
        alpha_bar_prev = extract(self.alphas_cumprod_prev, t_prev, x_t.shape)
        sigma = eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar)) * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
        mean_pred = (torch.sqrt(alpha_bar_prev) * pred_x_0 + torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * pred_eps)
        noise = torch.randn_like(x_t)
        pred_x_prev = mean_pred + nonzero_mask * sigma * noise
        return edict({"pred_x_prev": pred_x_prev, "pred_x_0": pred_x_0})

    @torch.no_grad()
    def sample(
        self,
        num,
        image_size=None,
        noise=None,
        classes=None,
        steps=None,
        clip_denoised=False,
        eta=0.0,
        verbose=True,
        **kwargs
    ):
        """
        Generate samples from the model using DDIM.
        
        Args:
            num: The number of samples to generate.
            image_size: The image size of the generated samples. If None, use default.
            noise: If specified, the noise from the encoder to sample.
                Should be of the same shape as `shape`.
            classes: The [N] tensor of class labels.
            steps: The number of steps to sample. If None, use the full number of steps.
            clip_denoised: If True, clip the denoised signal into [-1, 1].
            eta (float): The DDIM eta parameter.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        # in case backbone is ddp
        backbone = self.framework.backbone.module if hasattr(self.framework.backbone, "module") else self.framework.backbone
        backbone.eval()
        
        if image_size is None:
            image_size = backbone.image_size
        shape = (num, backbone.out_channels, image_size, image_size)

        device = backbone.device
        if noise is not None:
            img = noise
        else:
            img = torch.randn(shape, device=device)
        steps = steps if steps is not None else self.framework.timesteps
        jump = self.framework.timesteps // steps
        indices = list((jump * (i + 1), jump * i) for i in reversed(range(steps)))
        ret = edict({"samples": None, "pred_x_t": [], "pred_x_0": []})
        for t, t_prev in tqdm(indices, desc="DDIM Sampling", disable=not verbose):
            t = torch.tensor([t] * shape[0], device=device)
            t_prev = torch.tensor([t_prev] * shape[0], device=device)
            out = self.sample_once(img, t, t_prev, classes, clip_denoised, eta, **kwargs)
            img = out.pred_x_prev
            ret.pred_x_t.append(out.pred_x_prev)
            ret.pred_x_0.append(out.pred_x_0)
        ret.samples = img
        backbone.train()
        return ret

