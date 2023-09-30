import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm
from easydict import EasyDict as edict 

from .utils import extract


class DdpmSampler:
    """
    Generate samples from a diffusion model using DDPM schedule.

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

        ## calculations for diffusion q(x_t | x_0)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        ## calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        ### log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(np.append(self.posterior_variance[1], self.posterior_variance[1:]))
        self.posterior_mean_coef1 = (betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        self.posterior_mean_coef2 = ((1.0 - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - self.alphas_cumprod))

    def q_posterior_mean_variance(self, x_0, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:
            q(x_{t-1} | x_t, x_0)

        Args:
            x_0: The [N x C x ...] tensor of noiseless inputs.
            x_t: The [N x C x ...] tensor of noisy inputs.
            t: The [N] tensor of diffusion steps (minus 1). Here, 0 means one step.

        Returns:
            A tuple (mean, variance, log_variance), all of x_0's shape.
        """
        assert x_0.shape == x_t.shape, "x_0 and x_t must have the same shape"
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_0
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    @torch.no_grad()
    def p_mean_variance(self, x_t, t, classes=None, clip_denoised=False, **kwargs):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        Args:
            x_t: The [N x C x ...] tensor of noisy inputs at time t.
            t: The [N] tensor of diffusion steps (minus 1). Here, 0 means one step.
            classes: The [N] tensor of class labels.
            clip_denoised: If True, clip the denoised signal into [-1, 1].
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict with the following keys:
            - 'mean': the model mean output.
            - 'variance': the model variance output.
            - 'log_variance': the log of 'variance'.
            - 'pred_x_0': the prediction for x_0.
        """
        B, C = x_t.shape[:2]
        assert t.shape == (B,), "t must be a 1D tensor of shape (B,)"
        model_output = self.framework.model_inference(x_t, t, classes=classes, **kwargs)
        model_variance = extract(self.posterior_variance, t, x_t.shape)
        model_log_variance = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        pred_x_0 = self._predict_xstart_from_eps(x_t=x_t, t=t, eps=model_output)
        if clip_denoised:
            pred_x_0 = pred_x_0.clamp(-1, 1)
        model_mean, _, _ = self.q_posterior_mean_variance(x_0=pred_x_0, x_t=x_t, t=t)

        return edict({
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_x_0": pred_x_0,
            "pred_eps": model_output,
        })

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    @torch.no_grad()
    def sample_once(self, x_t, t, classes=None, clip_denoised=False, **kwargs):
        """
        Sample x_{t-1} from the model at the given timestep.
        
        Args:
            x_t: The [N x C x ...] tensor of noisy inputs at time t.
            t: The [N] tensor of diffusion steps (minus 1). Here, 0 means one step.
            classes: The [N] tensor of class labels.
            clip_denoised: If True, clip the denoised signal into [-1, 1].
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following keys:
            - 'pred_x_prev': x_{t-1}.
            - 'pred_x_0': a prediction of x_0.
        """
        out = self.p_mean_variance(x_t, t, classes, clip_denoised, **kwargs)
        noise = torch.randn_like(x_t)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
        pred_x_prev = out.mean + nonzero_mask * torch.exp(0.5 * out.log_variance) * noise
        return edict({"pred_x_prev": pred_x_prev, "pred_x_0": out.pred_x_0})

    @torch.no_grad()
    def sample(
        self,
        num,
        steps=None,
        image_size=None,
        noise=None,
        classes=None,
        clip_denoised=False,
        verbose=True,
        **kwargs
    ):
        """
        Generate samples from the model.

        Args:
            num: The number of samples to generate.
            image_size: The image size of the generated samples. If None, use default.
            noise: If specified, the noise from the encoder to sample.
                Should be of the same shape as `shape`.
            classes: The [N] tensor of class labels.
            clip_denoised: If True, clip the denoised signal into [-1, 1].
            verbose: If True, print the progress.
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
        indices = list(range(self.framework.timesteps))[::-1]
        ret = edict({"samples": None, "pred_x_t": [], "pred_x_0": []})
        for i in tqdm(indices, desc="DDPM Sampling", disable=not verbose):
            t = torch.tensor([i] * shape[0], device=device)
            out = self.sample_once(img, t, classes, clip_denoised, **kwargs)
            img = out.pred_x_prev
            ret.pred_x_t.append(out.pred_x_prev)
            ret.pred_x_0.append(out.pred_x_0)
        ret.samples = img
        backbone.train()
        return ret

