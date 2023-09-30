import os
import sys
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_path)
import glob
import argparse
import glm
import numpy as np
from PIL import Image
from tqdm import tqdm

import imageio
import rgbd_3d
from utils import colorize_depth, load_scene


if __name__ == '__main__':
    # Arguments and config
    parser = argparse.ArgumentParser()
    ## config
    parser.add_argument('--scene_dir', type=str, default='samples/imagenet128/viewset_3x9_steps_u1000_c50_guidance3.0', help='Scene directory')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory')
    parser.add_argument('--frames', type=int, default=60, help='Number of frames')
    parser.add_argument('--traj', type=str, default='swing', help='Camera trajectory')
    ## hyperparameters for 3d
    parser.add_argument('--atol', type=float, default=0.03, help='Absolute tolerance for meshing')
    parser.add_argument('--rtol', type=float, default=0.03, help='Relative tolerance for meshing')
    parser.add_argument('--erode_rgb', type=int, default=3, help='Erode rgb mask for meshing')
    opt = parser.parse_args()
    
    if opt.output_dir is None:
        opt.output_dir = opt.scene_dir

    os.makedirs(os.path.join(opt.output_dir, 'results'), exist_ok=True)
    os.makedirs(os.path.join(opt.output_dir, 'videos'), exist_ok=True)
    scenes = glob.glob(os.path.join(opt.scene_dir, 'scenes', '*.npz'))
    scenes.sort()
    print(f'Found {len(scenes)} scenes.')

    # camera trajectory
    if opt.traj == 'swing':
        traj_yaw = [0.6 * np.cos(t) for t in np.linspace(0, 2 * np.pi, opt.frames)]
        traj_pitch = [0.15 * np.sin(t) for t in np.linspace(0, 2 * np.pi, opt.frames)]
        modelviews = [glm.lookAt(
            glm.vec3(np.sin(yaw) * np.cos(pitch), np.sin(pitch), np.cos(yaw) * np.cos(pitch)),
            glm.vec3( 0.0, 0.0, 0.0),
            glm.vec3( 0.0, 1.0, 0.0)
        ) for yaw, pitch in zip(traj_yaw, traj_pitch)]
    elif opt.traj == 'random':
        modelviews = [[] for _ in range(len(scenes))]
        for i in range(len(scenes)):
            yaw = np.clip(0.3 * np.random.normal(), -0.6, 0.6)
            pitch = np.clip(0.15 * np.random.normal(), -0.15, 0.15)
            modelviews[i].append(glm.lookAt(
                glm.vec3(np.sin(yaw) * np.cos(pitch), np.sin(pitch), np.cos(yaw) * np.cos(pitch)),
                glm.vec3(0, 0, 0),
                glm.vec3(0, 1, 0),
            ))
    else:
        raise NotImplementedError
    
    # Render
    ssaa = 5
    ssaa_offset = ssaa // 2
    renderer = rgbd_3d.AggregationRenderer(128 * ssaa, 128, near=0.1, far=200, device=0)
    for i, scene in tqdm(enumerate(scenes), total=len(scenes), desc='Rendering', position=0):
        meshes, colors = load_scene(scene, atol=opt.atol, rtol=opt.rtol, erode_rgb=opt.erode_rgb)
        res = renderer.render(
            meshes, colors,
            modelviews[i] if isinstance(modelviews[0], list) else modelviews,
            verbose=True,
            tqdm_args={'position': 1, 'leave': False}
        )
        if opt.traj == 'random':
            img = Image.fromarray((res['color'] * 255).astype(np.uint8)).resize((128, 128), Image.Resampling.LANCZOS)
            imageio.imwrite(os.path.join(opt.output_dir, 'results', f'{os.path.basename(scene)[:-4]}.png'), img)
        else:
            colors = []
            depths = []
            for frame in res:
                colors.append(Image.fromarray((frame['color'] * 255).astype(np.uint8)).resize((128, 128), Image.Resampling.LANCZOS))
                depths.append((colorize_depth(
                    rgbd_3d.utils.project_depth(frame['depth'][ssaa_offset::ssaa, ssaa_offset::ssaa]), min=0, max=1
                ) * 255).astype(np.uint8))
            colors = np.stack(colors, axis=0)
            depths = np.stack(depths, axis=0)
            imageio.mimsave(os.path.join(opt.output_dir, 'videos', f'{os.path.basename(scene)[:-4]}.mp4'), colors, fps=30)
            imageio.mimsave(os.path.join(opt.output_dir, 'videos', f'{os.path.basename(scene)[:-4]}_depth.mp4'), depths, fps=30)

        