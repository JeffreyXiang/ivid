import os
import sys
import glob
import json
import argparse
from easydict import EasyDict as edict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist

from torchinfo import summary

import datasets
import diffusion.backbones as backbones
import diffusion.frameworks as frameworks
import diffusion.trainers as trainers


def find_latest_ckpt(cfg):
    # Load checkpoint
    cfg['load_ckpt'] = None
    if cfg.load_dir != '':
        if cfg.ckpt == 'latest':
            files = glob.glob(os.path.join(cfg.load_dir, 'ckpts', '*.pt'))
            if len(files) != 0:
                cfg.load_ckpt = max([
                    int(os.path.basename(f).split('step')[-1].split('.')[0])
                    for f in files
                ])
        elif cfg.ckpt == 'none':
            cfg.load_ckpt = None
        else:
            cfg.load_ckpt = int(cfg.ckpt)
    return cfg


def setup_dist(rank, local_rank, world_size, master_addr, master_port):
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(local_rank)
    torch.cuda.set_device(local_rank)
    dist.init_process_group('nccl', rank=rank, world_size=world_size)


def get_model_summary(model: nn.Module):
    model_summary = 'Parameters:\n'
    model_summary += '=' * 128 + '\n'
    model_summary += f'{"Name":<{72}}{"Shape":<{32}}{"Type":<{16}}{"Grad"}\n'
    for name, param in model.named_parameters():
        model_summary += f'{name:<{72}}{str(param.shape):<{32}}{str(param.dtype):<{16}}{param.requires_grad}\n'
    model_summary += '\n\nModel Summary:\n'
    assert hasattr(model, 'example_inputs'), 'Backbone must have attribute example_inputs for model summary'
    model_summary += str(summary(
        model,
        input_data = model.example_inputs,
        mode="train",
        col_names=("input_size", "output_size", "num_params", "mult_adds"),
        row_settings=("depth", "var_names"),
        verbose=0
    ))
    return model_summary


def main(local_rank, cfg):
    # Set up distributed training
    rank = cfg.node_rank * cfg.num_gpus + local_rank
    world_size = cfg.num_nodes * cfg.num_gpus
    setup_dist(rank, local_rank, world_size, cfg.master_addr, cfg.master_port)

    # Load data
    dataset = getattr(datasets, cfg.dataset.name)(cfg.data_dir, **cfg.dataset.args)
    if hasattr(cfg.backbone.args, 'num_classes') and cfg.backbone.args.num_classes == 'auto':
        cfg.backbone.args.num_classes = dataset.num_classes

    # Build model
    backbone = getattr(backbones, cfg.backbone.name)(**cfg.backbone.args).cuda()
    framework = getattr(frameworks, cfg.framework.name)(backbone, **cfg.framework.args)

    # Model summary
    if rank == 0:
        model_summary = get_model_summary(backbone)
        print('\n\n' + model_summary)
        with open(os.path.join(cfg.output_dir, 'model_summary.txt'), 'w') as fp:
            print(model_summary, file=fp)

    # Build trainer
    trainer = getattr(trainers, cfg.trainer.name)(framework, dataset, cfg.output_dir, **cfg.trainer.args)

    # Load checkpoint
    if cfg.load_ckpt is not None:
        trainer.load(cfg.load_ckpt)

    # Train
    trainer.run()


if __name__ == '__main__':
    # Check environment
    print('\n\nEnvironment:')
    print('=' * 80)
    print(f'NVIDIA_VISIBLE_DEVICES={os.environ.get("NVIDIA_VISIBLE_DEVICES", None)}')
    print(f'NVIDIA_DRIVER_CAPABILITIES={os.environ.get("NVIDIA_DRIVER_CAPABILITIES", None)}')
    print(f'LD_LIBRARY_PATH={os.environ.get("LD_LIBRARY_PATH", None)}')
    print('EGL libraries:')
    os.system('ldconfig -p | grep libEGL')
    print('EGL ICD files:')
    os.system('ls /usr/share/glvnd/egl_vendor.d')
    
    # Arguments and config
    parser = argparse.ArgumentParser()
    ## config
    parser.add_argument('--config', type=str, required=True, help='Experiment config file')
    ## io and resume
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--load_dir', type=str, default='', help='Load directory, default to output_dir')
    parser.add_argument('--ckpt', type=str, default='latest', help='Checkpoint step')
    parser.add_argument('--data_dir', type=str, default='./data/', help='Data directory')
    ## multi-node and multi-gpu
    parser.add_argument('--num_nodes', type=int, default=1, help='Number of nodes')
    parser.add_argument('--node_rank', type=int, default=0, help='Node rank')
    parser.add_argument('--num_gpus', type=int, default=-1, help='Number of GPUs per node, default to all')
    parser.add_argument('--master_addr', type=str, default='localhost', help='Master address for distributed training')
    parser.add_argument('--master_port', type=str, default='12345', help='Port for distributed training')
    opt = parser.parse_args()
    opt.load_dir = opt.load_dir if opt.load_dir != '' else opt.output_dir
    opt.num_gpus = torch.cuda.device_count() if opt.num_gpus == -1 else opt.num_gpus
    ## Load config
    config = json.load(open(opt.config, 'r'))
    ## Combine arguments and config
    cfg = edict()
    cfg.update(opt.__dict__)
    cfg.update(config)
    print('\n\nConfig:')
    print('=' * 80)
    print(json.dumps(cfg.__dict__, indent=4))

    # Prepare output directory
    if cfg.node_rank == 0:
        os.makedirs(cfg.output_dir, exist_ok=True)
        ## Save command and config
        with open(os.path.join(cfg.output_dir, 'command.txt'), 'w') as fp:
            print(' '.join(['python'] + sys.argv), file=fp)
        with open(os.path.join(cfg.output_dir, 'config.json'), 'w') as fp:
            json.dump(config, fp, indent=4)

    # Avoid IB insufficient pinned memory error
    # if cfg.num_nodes > 1:
    #     os.system('echo "* soft memlock unlimited" | sudo tee -a /etc/security/limits.conf')
    #     os.system('echo "* hard memlock unlimited" | sudo tee -a /etc/security/limits.conf')

    # Run
    cfg['load_ckpt'] = None
    cfg = find_latest_ckpt(cfg)
    if cfg.num_gpus > 1:
        mp.spawn(main, args=(cfg,), nprocs=cfg.num_gpus, join=True)
    else:
        main(0, cfg)
            