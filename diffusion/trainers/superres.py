import os
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np

from torchvision import utils

from .utils import *
from .basic import BasicTrainer


class SuperResTrainer(BasicTrainer):
    """
    Trainer for training a super-resolution model.
    
    Args:
        same as BasicTrainer
    """

    def __init__(self,
        framework,
        dataset,
        output_dir,
        *,
        max_steps,
        batch_size=None,
        batch_size_per_gpu=None,
        batch_split=None,
        learning_rate=1e-4,
        weight_decay=0.0,
        ema_rate=0.9999,
        fp16_mode='inflat_all',
        fp16_scale_growth=1e-3,
        finetune_ckpt=None,
        i_print=1000,
        i_log=500,
        i_sample=10000,
        i_save=10000,
        i_ddpcheck=10000,
    ):
        super().__init__(
            framework,
            dataset,
            output_dir,
            max_steps=max_steps,
            batch_size=batch_size,
            batch_size_per_gpu=batch_size_per_gpu,
            batch_split=batch_split,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            ema_rate=ema_rate,
            fp16_mode=fp16_mode,
            fp16_scale_growth=fp16_scale_growth,
            i_print=i_print,
            i_log=i_log,
            i_sample=i_sample,
            i_save=i_save,
            i_ddpcheck=i_ddpcheck,
        )
        self.finetune_ckpt = finetune_ckpt
        if self.finetune_ckpt is not None:
            self.finetune_from(self.finetune_ckpt)

    def finetune_from(self, finetune_ckpt):
        """
        Finetune from a checkpoint.
        Should be called by all processes.
        """
        if self.is_master:
            print(f'\nFinetuning from {finetune_ckpt} ...')

        model_ckpt = torch.load(read_file_dist(finetune_ckpt), map_location=self.device)
        model_state_dict = self.backbone.state_dict()
        for k, v in model_ckpt.items():
            if model_ckpt[k].shape != model_state_dict[k].shape:
                if self.is_master:
                    print(f'Warning: {k} shape mismatch, {model_ckpt[k].shape} vs {model_state_dict[k].shape}, padding with zeros.')
                weight = torch.zeros_like(model_state_dict[k])
                weight[:, :model_ckpt[k].shape[1]] = model_ckpt[k]
                model_ckpt[k] = weight
        self.backbone.load_state_dict(model_ckpt)
        self._state_dict_to_master_params(self.master_params, model_ckpt)
        if self.fp16_mode is not None:
            self.backbone.convert_to_fp16()

        dist.barrier()
        if self.is_master:
            print('Done.')

        self.check_ddp()

    @torch.no_grad()
    def sample(self, suffix=None, num_samples=9, batch_size=9):
        """
        Sample images from the diffusion model.
        Should be called only by the rank 0 process.
        """
        assert self.is_master, 'sample() should be called only by the rank 0 process.'
        print(f'\nSampling {num_samples} images...', end='')

        # Test data loader.
        test_loader = torch.utils.data.DataLoader(
            copy.deepcopy(self.dataset),
            batch_size=num_samples,
            shuffle=True,
            num_workers=0,
        )

        if suffix is None:
            suffix = f'step{self.step:07d}'

        all_images_list = []
        for i in range(0, num_samples, batch_size):
            batch = min(batch_size, num_samples - i)
            data = next(iter(test_loader))
            args = {k: v.cuda() for k, v in data.items()}
            res = self.sampler.sample(
                batch,
                **args,
                steps=50, strength=3.0, verbose=False
            )
            all_images_list.append(res.samples)
        all_images = torch.cat(all_images_list, dim = 0)
        utils.save_image(data['x_0'][:, :3], os.path.join(self.output_dir, 'samples', f'rgb_gt_{suffix}.png'), nrow=int(np.sqrt(num_samples)), normalize=True, value_range=(-1, 1))
        utils.save_image(data['y'][:, :3], os.path.join(self.output_dir, 'samples', f'rgb_cond_{suffix}.png'), nrow=int(np.sqrt(num_samples)), normalize=True, value_range=(-1, 1))
        utils.save_image(all_images[:, :3], os.path.join(self.output_dir, 'samples', f'rgb_{suffix}.png'), nrow=int(np.sqrt(num_samples)), normalize=True, value_range=(-1, 1))
        utils.save_image(data['x_0'][:, 3:], os.path.join(self.output_dir, 'samples', f'depth_gt_{suffix}.png'), nrow=int(np.sqrt(num_samples)), normalize=True, value_range=(-1, 1))
        utils.save_image(data['y'][:, 3:], os.path.join(self.output_dir, 'samples', f'depth_cond_{suffix}.png'), nrow=int(np.sqrt(num_samples)), normalize=True, value_range=(-1, 1))
        utils.save_image(all_images[:, 3:], os.path.join(self.output_dir, 'samples', f'depth_{suffix}.png'), nrow=int(np.sqrt(num_samples)), normalize=True, value_range=(-1, 1))
        print(' Done.')
