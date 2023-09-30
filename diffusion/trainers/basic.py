import os
import time
import json
import copy
from tqdm import tqdm
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np

from torchvision import utils
import mlflow

from .utils import *
from .. import samplers, frameworks


class BasicTrainer:
    """
    Basic trainer for training a diffusion model.
    
    Args:
        framework (nn.Module): Framework model.
        dataset (torch.utils.data.Dataset): Dataset.
        batch_size (int): Batch size.
        batch_size_per_gpu (int): Batch size per GPU. If specified, batch_size will be ignored.
        batch_split (int): Split batch with gradient accumulation.
        max_steps (int): Max steps.
        learning_rate (float): Learning rate.
        weight_decay (float): Weight decay.
        ema_rate (float or list): Exponential moving average rates.
        fp16_mode (str): FP16 mode.
            - None: No FP16.
            - 'inflat_all': Hold a inflated fp32 master param for all params.
        fp16_scale_growth (float): Scale growth for FP16 gradient backpropagation.
        output_dir (str): Output directory.
        i_print (int): Print interval.
        i_log (int): Log interval.
        i_sample (int): Sample interval.
        i_save (int): Save interval.
        i_ddpcheck (int): DDP check interval.
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
        i_print=1000,
        i_log=500,
        i_sample=10000,
        i_save=10000,
        i_ddpcheck=10000,
    ):
        assert batch_size is not None or batch_size_per_gpu is not None, 'Either batch_size or batch_size_per_gpu must be specified.'

        self.backbone = framework.backbone
        self.framework = framework
        self.dataset = dataset
        self.batch_size = batch_size if batch_size_per_gpu is None else batch_size_per_gpu * dist.get_world_size()
        self.batch_size_per_gpu = batch_size_per_gpu if batch_size_per_gpu is not None else batch_size // dist.get_world_size()
        self.batch_split = batch_split if batch_split is not None else 1
        self.max_steps = max_steps
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.ema_rate = [ema_rate] if isinstance(ema_rate, float) else ema_rate
        self.fp16_mode = fp16_mode
        self.fp16_scale_growth = fp16_scale_growth
        self.log_scale = 20.0
        self.device = self.backbone.device

        assert self.batch_size % dist.get_world_size() == 0, 'Batch size must be divisible by the number of GPUs.'
        assert self.batch_size_per_gpu % self.batch_split == 0, 'Batch size per GPU must be divisible by batch split.'

        self.output_dir = output_dir
        self.i_print = i_print
        self.i_log = i_log
        self.i_sample = i_sample
        self.i_save = i_save
        self.i_ddpcheck = i_ddpcheck
        self.sampler = samplers.DdimSampler(self.framework)

        os.makedirs(os.path.join(self.output_dir, 'ckpts'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'samples'), exist_ok=True)

        # Multi-GPU params
        assert dist.is_initialized(), 'Only support distributed training. torch.distributed is not initialized.'
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.local_rank = dist.get_rank() % torch.cuda.device_count()
        self.is_master = self.rank == 0
        assert self.device == torch.device(f'cuda:{self.local_rank}'), 'Device must be cuda:local_rank.'
    
        self.step = 0
        self.model_params = [p for p in self.backbone.parameters()]
        self.model_param_names = [n for n, p in self.backbone.named_parameters()]

        # Check if the model contains fp16 params
        self.fp16_mode = self.fp16_mode if any([p.dtype == torch.float16 for p in self.model_params]) else None

        # Prepare distributed data parallel
        framework.backbone = DDP(
            self.backbone,
            device_ids=[self.local_rank],
            output_device=self.local_rank,
            broadcast_buffers=False,
            bucket_cap_mb=128,
            find_unused_parameters=False
        )

        # Build master params
        if self.fp16_mode == 'inflat_all':
            self.master_params = make_master_params(self.model_params)
        elif self.fp16_mode is None:
            self.master_params = self.model_params
        else:
            raise NotImplementedError(f'FP16 mode {self.fp16_mode} is not implemented.')

        # Build EMA params
        if self.is_master:
            self.ema_params = [copy.deepcopy(self.master_params) for _ in self.ema_rate]

        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(self.master_params, lr=self.learning_rate, weight_decay=self.weight_decay)

        # Prepare dataloader
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size_per_gpu,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
            sampler=torch.utils.data.distributed.DistributedSampler(
                self.dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
            )
        )
        self.dataloader = cycle(self.dataloader)

        self.check_ddp()

        if self.is_master:
            print('\n\nTrainer initialized.')
            print(f'  - Backbone: {self.backbone.__class__.__name__}')
            print(f'  - Framework: {self.framework.__class__.__name__}')
            print(f'  - Dataset: {self.dataset.__class__.__name__}')
            print(f'  - Image size: {self.dataset.image_size}')
            if hasattr(self.dataset, 'num_classes'):
                print(f'  - Number of classes: {self.dataset.num_classes}')
            print(f'  - Number of steps: {self.max_steps}')
            print(f'  - Number of GPUs: {self.world_size}')
            print(f'  - Batch size: {self.batch_size}')
            print(f'  - Batch size per GPU: {self.batch_size_per_gpu}')
            print(f'  - Batch split: {self.batch_split}')
            print(f'  - Learning rate: {self.learning_rate}')
            print(f'  - Weight decay: {self.weight_decay}')
            print(f'  - EMA rate: {self.ema_rate}')
            print(f'  - FP16 mode: {self.fp16_mode}')
            print(f'  - FP16 scale growth: {self.fp16_scale_growth}')

    def _master_params_to_state_dict(self, master_params):
        """
        Convert master params to a state_dict.
        """
        if self.fp16_mode == 'inflat_all':
            master_params = unflatten_master_params(self.model_params, master_params)
        state_dict = self.backbone.state_dict()
        for i, name in enumerate(self.model_param_names):
            state_dict[name] = master_params[i]
        return state_dict

    def _state_dict_to_master_params(self, master_params, state_dict):
        """
        Convert a state_dict to master params.
        """
        params = [state_dict[name] for name in self.model_param_names]
        if self.fp16_mode == 'inflat_all':
            model_params_to_master_params(params, master_params)

    def load(self, load_dir, step=0):
        """
        Load a checkpoint.
        Should be called by all processes.
        """
        if self.is_master:
            print(f'\nLoading checkpoint from step {step}...', end='')

        model_ckpt = torch.load(read_file_dist(os.path.join(load_dir, 'ckpts', f'model_step{step:07d}.pt')), map_location=self.device)
        self.backbone.load_state_dict(model_ckpt)
        self._state_dict_to_master_params(self.master_params, model_ckpt)
        if self.fp16_mode is not None:
            self.backbone.convert_to_fp16()
        del model_ckpt

        if self.is_master:
            for i, ema_rate in enumerate(self.ema_rate):
                ema_ckpt = torch.load(os.path.join(load_dir, 'ckpts', f'ema_{ema_rate}_step{step:07d}.pt'), map_location=self.device)
                self._state_dict_to_master_params(self.ema_params[i], ema_ckpt)
                del ema_ckpt
        
        misc_ckpt = torch.load(read_file_dist(os.path.join(load_dir, 'ckpts', f'misc_step{step:07d}.pt')), map_location=torch.device('cpu'))
        self.optimizer.load_state_dict(misc_ckpt['optimizer'])
        self.step = misc_ckpt['step']
        if self.fp16_mode is not None:
            self.log_scale = misc_ckpt['log_scale']
        del misc_ckpt

        dist.barrier()
        if self.is_master:
            print(' Done.')

        self.check_ddp()

    def save(self):
        """
        Save a checkpoint.
        Should be called only by the rank 0 process.
        """
        assert self.is_master, 'save() should be called only by the rank 0 process.'
        print(f'\nSaving checkpoint at step {self.step}...', end='')

        model_ckpt = self._master_params_to_state_dict(self.master_params)
        torch.save(model_ckpt, os.path.join(self.output_dir, 'ckpts', f'model_step{self.step:07d}.pt'))

        for i, ema_rate in enumerate(self.ema_rate):
            ema_ckpt = self._master_params_to_state_dict(self.ema_params[i])
            torch.save(ema_ckpt, os.path.join(self.output_dir, 'ckpts', f'ema_{ema_rate}_step{self.step:07d}.pt'))

        misc_ckpt = {
            'optimizer': self.optimizer.state_dict(),
            'step': self.step,
        }
        if self.fp16_mode is not None:
            misc_ckpt['log_scale'] = self.log_scale
        torch.save(misc_ckpt, os.path.join(self.output_dir, 'ckpts', f'misc_step{self.step:07d}.pt'))
        print(' Done.')

    @torch.no_grad()
    def sample(self, suffix=None, num_samples=25, batch_size=25):
        """
        Sample images from the diffusion model.
        Should be called only by the rank 0 process.
        """
        assert self.is_master, 'sample() should be called only by the rank 0 process.'
        print(f'\nSampling {num_samples} images...', end='')

        if suffix is None:
            suffix = f'step{self.step:07d}'

        all_images_list = []
        for i in range(0, num_samples, batch_size):
            batch = min(batch_size, num_samples - i)
            if self.backbone.num_classes is not None:
                classes = torch.randint(0, self.backbone.num_classes, (batch,), device=self.device)
            else:
                classes = None
            if isinstance(self.framework, frameworks.ClassifierFreeGuidance):
                res = self.sampler.sample(batch, classes=classes, steps=250, strength=3.0, verbose=False)
            else:
                res = self.sampler.sample(batch, classes=classes, steps=250, verbose=False)
            all_images_list.append(res.samples)
        all_images = torch.cat(all_images_list, dim = 0)
        utils.save_image(all_images[:, :3], os.path.join(self.output_dir, 'samples', f'rgb_{suffix}.png'), nrow=int(np.sqrt(num_samples)), normalize=True, value_range=(-1, 1))
        if self.backbone.out_channels == 4:
            utils.save_image(all_images[:, 3:], os.path.join(self.output_dir, 'samples', f'depth_{suffix}.png'), nrow=int(np.sqrt(num_samples)), normalize=True, value_range=(-1, 1))
        print(' Done.')

    def update_ema(self):
        """
        Update exponential moving average.
        Should only be called by the rank 0 process.
        """
        assert self.is_master, 'update_ema() should be called only by the rank 0 process.'
        for i, ema_rate in enumerate(self.ema_rate):
            for master_param, ema_param in zip(self.master_params, self.ema_params[i]):
                ema_param.detach().mul_(ema_rate).add_(master_param, alpha=1.0 - ema_rate)

    def check_ddp(self):
        """
        Check if DDP is working properly.
        Should be called by all process.
        """
        if self.is_master:
            print('\nPerforming DDP check...')

        # check self.model_params
        if self.is_master:
            print('Checking if stored parameters list is consistent with model parameters...')
        dist.barrier()
        assert all([p1 is p2 for p1, p2 in zip(
            self.model_params,
            [p for p in self.backbone.parameters()]
        )]), 'self.model_params is not coherent with self.backbone.parameters()'

        # check self.master_params
        if self.is_master:
            print('Checking if optimizer parameters list is consistent with master parameters...')
        dist.barrier()
        assert all([p1 is p2 for p1, p2 in zip(
            self.master_params,
            [p for p in self.optimizer.param_groups[0]['params']]
        )]), 'self.master_params is not coherent with self.optimizer.param_groups[0][\'params\']'

        if self.is_master:
            print('Checking if parameters are consistent across processes...')
        dist.barrier()
        for p in self.master_params:
            # split to avoid OOM
            for i in range(0, p.numel(), 10000000):
                sub_size = min(10000000, p.numel() - i)
                sub_p = p.detach().view(-1)[i:i+sub_size]
                # gather from all processes
                sub_p_gather = [torch.empty_like(sub_p) for _ in range(self.world_size)]
                dist.all_gather(sub_p_gather, sub_p)
                # check if equal
                assert all([torch.equal(sub_p, sub_p_gather[i]) for i in range(self.world_size)]), 'parameters are not consistent across processes'

        dist.barrier()
        if self.is_master:
            print('Done.')

    def run_step(self):
        """
        Run a training step.
        """
        zero_grad(self.model_params)

        # Load data
        data = next(self.dataloader)
        for key in data.keys():
            data[key] = data[key].to(self.device).chunk(self.batch_split) if self.batch_split > 1 else [data[key].to(self.device)]
        data_list = []
        for i in range(self.batch_split):
            data_list.append({key: data[key][i] for key in data.keys()})

        # Forward and backward
        for i, mb_data in enumerate(data_list):
            # sync at the end of each batch split
            sync_context = self.framework.backbone.no_sync if i != len(data_list) - 1 else nullcontext
            with sync_context():
                losses = self.framework.training_losses(**mb_data)
                loss = losses['loss']
                if self.fp16_mode is not None:
                    loss = loss * (2 ** self.log_scale)
                loss.backward()

        # Optimize
        if self.fp16_mode is not None:
            # if any gradient is NaN, reduce the loss scale
            if any(not p.grad.isfinite().all() for p in self.model_params):
                self.log_scale -= 1
                return
            model_grads_to_master_grads(self.model_params, self.master_params)
            self.master_params[0].grad.mul_(1.0 / (2 ** self.log_scale))
        self.optimizer.step()
        if self.fp16_mode is not None:
            master_params_to_model_params(self.model_params, self.master_params)
            self.log_scale += self.fp16_scale_growth

        # Update exponential moving average
        if self.is_master:
            self.update_ema()

        return losses

    def run(self):
        """
        Run training.
        """
        if self.is_master:
            print('\nStarting training...')
            log_file = open(os.path.join(self.output_dir, 'log.txt'), 'a')
            if self.step == 0:
                self.sample(suffix='init')

        log = []
        time_elapsed = 0.0
        with tqdm(total=self.max_steps, initial=self.step, disable=not self.is_master, ncols=80, desc='Training') as pbar:
            while self.step < self.max_steps:

                time_start = time.time()

                losses = self.run_step()

                time_end = time.time()
                time_elapsed += time_end - time_start

                self.step += 1

                if self.step % self.i_print == 0:
                    pbar.update(self.step - pbar.n)

                # Check ddp
                if self.i_ddpcheck is not None and self.step % self.i_ddpcheck == 0:
                    self.check_ddp()

                if self.is_master:
                    log.append((self.step, {}))

                    # Log time
                    log[-1][1]['time'] = {
                        'step': time_end - time_start,
                        'elapsed': time_elapsed,
                    }

                    # Log losses
                    if losses is not None:
                        log[-1][1]['loss'] = {k: v.item() for k, v in losses.items()}
                    else:
                        log[-1][1]['loss'] = None

                    # Log scale
                    if self.fp16_mode is not None:
                        log[-1][1]['log_scale'] = self.log_scale

                    if self.step % self.i_log == 0:
                        log_str = '\n'.join([
                            f'{step}: {json.dumps(log)}' for step, log in log
                        ])
                        print(log_str, file=log_file)
                        log_file.flush()
                        mlflow.log_metric('step_time', np.mean([
                            log['time']['step'] for _, log in log
                        ]), self.step)
                        mlflow.log_metric('loss', np.mean([
                            log['loss']['loss'] for _, log in log if log['loss'] is not None
                        ]), self.step)
                        mlflow.log_metric('log_scale', self.log_scale, self.step)
                        log = []

                    # Save checkpoint
                    if self.step % self.i_save == 0:
                        self.save()

                    # Sample images
                    if self.step % self.i_sample == 0:
                        self.sample()
