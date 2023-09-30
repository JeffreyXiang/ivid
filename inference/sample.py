import os
import sys
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_path)
import json
import argparse
from easydict import EasyDict as edict
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import numpy as np
import glm

from torchvision import utils
import imageio
import threading

import diffusion.backbones as backbones
import diffusion.frameworks as frameworks
import diffusion.samplers as samplers

import rgbd_3d
from utils import parse_int_list, colorize_depth, reorder, save_scene


@torch.no_grad()
def sample_all(
        framework_uncond,
        framework_cond,
        seeds_or_num_samples,
        steps_uncond,
        steps_cond,
        modelviews,
        fov=45,
        near=0.6,
        far=5,
        atol=0.03,
        rtol=0.03,
        erode_rgb=2,
        classes=None,
        guidance=3.0,
        batchsize=10,
    ):
    sampler_uncond = samplers.DdimSampler(framework_uncond) if steps_uncond < 1000 else samplers.DdpmSampler(framework_uncond)
    if framework_cond is not None:
        sampler_cond = samplers.DdimSampler(framework_cond)
        renderer = [rgbd_3d.AggregationRenderer(128 * 3, 128) for _ in range(batchsize)]
    num_samples = seeds_or_num_samples if not isinstance(seeds_or_num_samples, list) else len(seeds_or_num_samples)
    seeds = seeds_or_num_samples if isinstance(seeds_or_num_samples, list) else None

    with tqdm(total=num_samples, position=0) as pbar:
        for i in range(0, num_samples, batchsize):
            pbar.set_description(f'Seed {seeds[i]}' if seeds is not None else 'Sample')
            bs = min(batchsize, num_samples - i)
            meshes = [[] for _ in range(bs)]
            colors = [[] for _ in range(bs)]
            samples = []
            conds = {'color': [], 'depth': []}

            # inputs
            if seeds is not None:
                noise = []
                for j in range(bs):
                    torch.manual_seed(seeds[i + j]) 
                    noise.append(torch.randn(1, 4, 128, 128, device='cuda'))
                noise = torch.cat(noise, dim=0)
            else:
                noise = None
            b_classes = torch.tensor(classes[i: i+bs]).long().cuda() if classes is not None else None

            s_modelviews = modelviews[i] if isinstance(modelviews[0], list) else modelviews
            for j, modelview in enumerate(tqdm(s_modelviews, position=1, desc='Views', leave=False)):
                if j == 0:
                    # Sample unconditional
                    if isinstance(framework_uncond, frameworks.ClassifierFreeGuidance):
                        res = sampler_uncond.sample(bs, noise=noise, classes=b_classes, steps=steps_uncond, strength=guidance, verbose=False)
                    else:
                        res = sampler_uncond.sample(bs, noise=noise, classes=b_classes, steps=steps_uncond, verbose=False)
                    samples.append(res.samples.clone().detach())
                    rgbd = res.samples.clone().detach().cpu().numpy().transpose(0, 2, 3, 1) * 0.5 + 0.5
                    del res
                else:
                    # Sample conditional
                    cond = [rgbd_3d.utils.aggregate_conditions(
                        renderer[k],
                        meshes[k],
                        colors[k],
                        s_modelviews[j],
                        fov=fov,
                        near=near,
                        far=far,
                        atol=atol,
                        rtol=rtol,
                        erode_rgb=erode_rgb,
                    ) for k in range(bs)]
                    cond = edict({k: np.stack([c[k] for c in cond], axis=0) for k in cond[0].keys()})
                    conds['color'].append(torch.from_numpy(cond.color).permute(0, 3, 1, 2).float() * 2 - 1)
                    conds['depth'].append(torch.from_numpy(cond.depth).permute(0, 3, 1, 2).float() * 2 - 1)
                    args = {
                        'y': torch.from_numpy(np.concatenate([cond.color, cond.depth], axis=-1)).permute(0, 3, 1, 2).float().cuda() * 2 - 1,
                        'mask': torch.from_numpy(cond.mask).permute(0, 3, 1, 2).float().cuda(),
                        'mask_rgb': torch.from_numpy(cond.mask_rgb).permute(0, 3, 1, 2).float().cuda(),
                        'replace_rgb': (
                            0.1,
                            torch.from_numpy(cond.color).permute(0, 3, 1, 2).float().cuda() * 2 - 1,
                            torch.from_numpy(cond.mask_rgb).permute(0, 3, 1, 2).float().cuda()
                        ),
                        'replace_depth': (
                            0.2,
                            torch.from_numpy(cond.depth).permute(0, 3, 1, 2).float().cuda() * 2 - 1,
                            torch.from_numpy(cond.mask).permute(0, 3, 1, 2).float().cuda()
                        ),
                        'constrain_depth': (
                            0.5,
                            torch.from_numpy(cond.depth_convex).permute(0, 3, 1, 2).float().cuda() * 2 - 1,
                        ),
                    }
                    if isinstance(framework_uncond, frameworks.ClassifierFreeGuidance):
                        res = sampler_cond.sample(bs, classes=b_classes, steps=steps_cond, strength=guidance, **args, verbose=False)
                    else:
                        res = sampler_cond.sample(bs, classes=b_classes, steps=steps_cond, **args, verbose=False)
                    samples.append(res.samples.clone().detach())
                    rgbd = res.samples.clone().detach().cpu().numpy().transpose(0, 2, 3, 1) * 0.5 + 0.5
                    del res
                for k in range(bs):
                    meshes[k].append(rgbd_3d.utils.depth_to_mesh(
                        rgbd_3d.utils.linearize_depth(rgbd[k, :, :, 3:], near, far),
                        padding='frustum',
                        fov = fov,
                        modelview=modelview,
                        atol=atol,
                        rtol=rtol,
                        erode_rgb=erode_rgb,
                        cal_normal=True,
                    ))
                    colors[k].append(rgbd[k, :, :, :3])

            pbar.update(bs)

            samples = torch.stack(samples, dim=1)
            conds = {k: torch.stack(v, dim=1) for k, v in conds.items()} if len(conds['color']) > 0 else None

            for j in range(bs):
                yield meshes[j], colors[j], samples[j], {k: v[j] for k, v in conds.items()} if conds is not None else None


def async_save(meshes, colors, samples, conds, suffix, cfg):
    def worker(meshes, colors, samples, conds, suffix, cfg):
        for _ in range(10):
            try:
                if cfg.viewset == 'uncond':
                    imageio.imwrite(os.path.join(cfg.output_dir, 'results', f'rgb_{suffix}.png'), (np.clip(samples[0, :3].cpu().numpy().transpose(1, 2, 0) * 0.5 + 0.5, 0, 1) * 255).astype(np.uint8))
                    save_scene(os.path.join(cfg.output_dir, 'scenes', f'scene_{suffix}.npz'), meshes, colors)
                elif cfg.viewset == 'random':
                    utils.save_image(samples[:, :3], os.path.join(cfg.output_dir, 'grids', f'rgb_{suffix}.png'), nrow=2, normalize=True, value_range=(-1, 1))
                    imageio.imwrite(os.path.join(cfg.output_dir, 'conds', f'rgb_{suffix}.png'), (np.clip(samples[0, :3].cpu().numpy().transpose(1, 2, 0) * 0.5 + 0.5, 0, 1) * 255).astype(np.uint8))
                    imageio.imwrite(os.path.join(cfg.output_dir, 'results', f'rgb_{suffix}.png'), (np.clip(samples[1, :3].cpu().numpy().transpose(1, 2, 0) * 0.5 + 0.5, 0, 1) * 255).astype(np.uint8))
                elif cfg.viewset == '3x9':
                    utils.save_image(reorder(samples[:, :3], cfg.viewset), os.path.join(cfg.output_dir, 'grids', f'rgb_{suffix}.png'), nrow=9, normalize=True, value_range=(-1, 1))
                    utils.save_image(reorder(colorize_depth(samples[:, 3:]), cfg.viewset), os.path.join(cfg.output_dir, 'grids', f'depth_{suffix}.png'), nrow=9, normalize=True, value_range=(-1, 1))
                    utils.save_image(reorder(conds['color'][:, :3], cfg.viewset), os.path.join(cfg.output_dir, 'conds', f'rgb_cond_{suffix}.png'), nrow=9, normalize=True, value_range=(-1, 1))    
                    utils.save_image(reorder(colorize_depth(conds['depth']), cfg.viewset), os.path.join(cfg.output_dir, 'conds', f'depth_cond_{suffix}.png'), nrow=9, normalize=True, value_range=(-1, 1))        
                    save_scene(os.path.join(cfg.output_dir, 'scenes', f'scene_{suffix}.npz'), meshes, colors)
                else:
                    raise NotImplementedError
                break
            except Exception as e:
                print(e)
                pass

    thread = threading.Thread(target=worker, args=(meshes, colors, samples, conds, suffix, cfg))
    thread.start()
    return thread


def main(rank, world_size, seeds, modelviews, classes, cfg, cfg_uncond, cfg_cond):
    torch.cuda.set_device(rank)

    # Build unconditional model
    backbone_uncond = getattr(backbones, cfg_uncond.backbone.name)(**cfg_uncond.backbone.args).cuda()
    framework_uncond = getattr(frameworks, cfg_uncond.framework.name)(backbone_uncond, **cfg_uncond.framework.args)
    ## Load checkpoint
    ckpt = torch.load(cfg.ckpt_uncond, map_location='cpu')
    backbone_uncond.load_state_dict(ckpt)

    # Build conditional model
    if cfg.viewset != 'uncond':
        backbone_cond = getattr(backbones, cfg_cond.backbone.name)(**cfg_cond.backbone.args).cuda()
        framework_cond = getattr(frameworks, cfg_cond.framework.name)(backbone_cond, **cfg_cond.framework.args)
        ## Load checkpoint
        ckpt = torch.load(cfg.ckpt_cond, map_location='cpu')
        backbone_cond.load_state_dict(ckpt)
    else:
        framework_cond = None

    seeds = seeds[rank::world_size] if seeds is not None else None
    idx = np.arange(cfg.num_samples)[rank::world_size] if cfg.num_samples is not None else None
    classes = classes[rank::world_size] if classes is not None else None
    modelviews = modelviews[rank::world_size] if isinstance(modelviews[0], list) else modelviews

    sample_process = sample_all(
        framework_uncond,
        framework_cond,
        seeds if seeds is not None else len(idx),
        cfg.steps_uncond,
        cfg.steps_cond,
        modelviews,
        classes=classes,
        guidance=cfg.guidance,
        batchsize=cfg.batchsize,
        fov=cfg.fov,
        near=cfg.near,
        far=cfg.far,
        atol=cfg.atol,
        rtol=cfg.rtol,
        erode_rgb=cfg.erode_rgb,
    )

    threads = []
    for i, (meshes, colors, samples, conds) in enumerate(sample_process):
        # Save
        suffix = []
        if classes is not None:
            suffix.append(f'class{classes[i]:03d}')
        if seeds is not None:
            suffix.append(f'seed{seeds[i]:05d}')
        else:
            suffix.append(f'{idx[i]:05d}')
        suffix = '_'.join(suffix)
        threads.append(async_save(meshes, colors, samples, conds, suffix, cfg))

    # Wait for all processes to complete
    for thread in threads:
        thread.join()


if __name__ == '__main__':
    # Arguments and config
    parser = argparse.ArgumentParser()
    ## config
    parser.add_argument('--config_uncond', type=str, default='configs/rgbd_imagenet_adm_128_large_cfg.json', help='Config file for unconditional model')
    parser.add_argument('--config_cond', type=str, default='configs/rgbd_imagenet_adm_128_large_cond.json', help='Config file for conditional model')
    parser.add_argument('--ckpt_uncond', type=str, default='ckpts/imagenet128_uncond.pt', help='Path to the checkpoint of unconditional model')
    parser.add_argument('--ckpt_cond', type=str, default='ckpts/imagenet128_cond.pt', help='Path to the checkpoint of conditional model')
    parser.add_argument('--output_dir', type=str, default='samples/imagenet128', help='Output directory')
    parser.add_argument('--seeds', type=str, default='0-8', help='Seeds for the generated samples')
    parser.add_argument('--num_samples', type=int, default=None, help='Number of samples, if specified, seeds will be ignored')
    parser.add_argument('--classes', type=str, default='mod', help='Classes for the generated samples')
    parser.add_argument('--viewset', type=str, default='3x9', help='Viewset for the iterative sampling process')
    parser.add_argument('--steps_uncond', type=int, default=1000, help='Number of sampling steps for unconditional model')
    parser.add_argument('--steps_cond', type=int, default=50, help='Number of sampling steps for conditional model')
    parser.add_argument('--guidance', type=float, default=3.0, help='Classfiier-free guidance strength')
    parser.add_argument('--batchsize', type=int, default=10, help='Batch size for inference')
    ## hyperparameters for 3d
    parser.add_argument('--fov', type=float, default=45, help='Field of view')
    parser.add_argument('--near', type=float, default=0.6, help='Near plane')
    parser.add_argument('--far', type=float, default=5, help='Far plane')
    parser.add_argument('--atol', type=float, default=0.03, help='Absolute tolerance for meshing')
    parser.add_argument('--rtol', type=float, default=0.03, help='Relative tolerance for meshing')
    parser.add_argument('--erode_rgb', type=int, default=3, help='Erode rgb mask for meshing')
    opt = parser.parse_args()
    ## Load config
    with open(opt.config_uncond, 'r') as fp:
        config_uncond = json.load(fp)
    with open(opt.config_cond, 'r') as fp:
        config_cond = json.load(fp)
    ## Combine arguments and config
    cfg_uncond = edict(config_uncond)
    cfg_cond = edict(config_cond)
    cfg = edict()
    cfg.update(opt.__dict__)

    # Prepare output directory
    cfg.output_dir = os.path.join(cfg.output_dir, f'viewset_{cfg.viewset}_steps_u{cfg.steps_uncond}_c{cfg.steps_cond}_guidance{cfg.guidance}')
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(os.path.join(cfg.output_dir, 'scenes'), exist_ok=True)
    os.makedirs(os.path.join(cfg.output_dir, 'conds'), exist_ok=True)
    os.makedirs(os.path.join(cfg.output_dir, 'grids'), exist_ok=True)
    os.makedirs(os.path.join(cfg.output_dir, 'results'), exist_ok=True)

    # Sample
    if cfg.num_samples is not None:
        num_samples = cfg.num_samples
        seeds = None
    else:        
        seeds = parse_int_list(cfg.seeds)
        num_samples = len(seeds)

    classes = None
    num_classes = cfg_uncond.backbone.args.num_classes
    if num_classes is not None:
        if cfg.classes == 'mod':
            classes = [seeds[i] % num_classes for i in range(num_samples)]
        elif cfg.classes == 'random':
            classes = [np.random.randint(num_classes) for i in range(num_samples)]
        elif cfg.classes == 'uniform':
            classes = [i % num_classes for i in range(num_samples)]
        else:
            classes = parse_int_list(cfg.classes)
    
    if cfg.viewset == 'uncond':
        modelviews = [glm.lookAt(
            glm.vec3(0, 0, 1),
            glm.vec3(0, 0, 0),
            glm.vec3(0, 1, 0),
        )]
    elif cfg.viewset == 'random':
        modelviews = [[glm.lookAt(
            glm.vec3(0, 0, 1),
            glm.vec3(0, 0, 0),
            glm.vec3(0, 1, 0),
        )] for _ in range(num_samples)]
        for i in range(num_samples):
            yaw = 0.3 * np.random.normal()
            pitch = 0.15 * np.random.normal()
            modelviews[i].append(glm.lookAt(
                glm.vec3(np.sin(yaw) * np.cos(pitch), np.sin(pitch), np.cos(yaw) * np.cos(pitch)),
                glm.vec3(0, 0, 0),
                glm.vec3(0, 1, 0),
            ))
    elif cfg.viewset == '3x9':
        yaws = [0.0]
        pitches = [0.0]
        for i in range(4): yaws += [(i + 1) * 0.15, -(i + 1) * 0.15]
        for i in range(1): pitches += [(i + 1) * 0.15, -(i + 1) * 0.15]
        modelviews = []
        for yaw in yaws:    
            for pitch in pitches:
                modelviews.append(glm.lookAt(
                    glm.vec3(np.sin(yaw) * np.cos(pitch), np.sin(pitch), np.cos(yaw) * np.cos(pitch)),
                    glm.vec3( 0.0, 0.0, 0.0),
                    glm.vec3( 0.0, 1.0, 0.0)
                ))
    else:
        raise NotImplementedError

    world_size = torch.cuda.device_count()
    if world_size == 1:
        main(0, 1, seeds, modelviews, classes, cfg, cfg_uncond, cfg_cond)
    else:
        mp.spawn(
            main,
            args=(world_size, seeds, modelviews, classes, cfg, cfg_uncond, cfg_cond),
            nprocs=world_size,
        )
    