import os
import sys
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_path)
from easydict import EasyDict as edict
import cv2
import numpy as np
import torch
import io
import imageio
import rgbd_3d

def parse_int_list(int_list_str):
    int_list_strs = int_list_str.split(',')
    ints = []
    for s in int_list_strs:
        if '-' in s:
            start, end = s.split('-')
            ints += list(range(int(start), int(end)+1))
        else:
            ints.append(int(s))
    return ints


def colorize_depth(depth, min=-1, max=1):
    is_tensor = isinstance(depth, torch.Tensor)
    if is_tensor:
        depth = depth.clone().detach().cpu().numpy()
    depth = depth.squeeze()
    if len(depth.shape) == 2:
        depth = depth[None]
    depth = (depth - min) / (max - min)
    depth = np.clip(1 - depth, 0, 1)
    colorized = []
    for i in range(depth.shape[0]):
        colorized.append(cv2.cvtColor(cv2.applyColorMap((depth[i]*255).astype(np.uint8), cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB))
    colorized = np.stack(colorized, axis=0) / 255
    if is_tensor:
        colorized = torch.from_numpy(colorized).permute(0, 3, 1, 2).float()
    colorized = colorized * (max - min) + min
    return colorized.squeeze()


def reorder(data, order='3x9'):
    data = list(data)
    if order == '3x9':
        if len(data) == 26:
            data.insert(0, -torch.ones_like(data[0]))
        order = [23, 17, 11, 5, 2, 8, 14, 20, 26,
                 21, 15, 9, 3, 0, 6, 12, 18, 24,
                 22, 16, 10, 4, 1, 7, 13, 19, 25]
        data = [data[i] for i in order]
        return torch.stack(data, dim=0)
    else:
        raise NotImplementedError


def collect_data(dataset, seeds, device='cuda'):
    data = edict()
    for seed in seeds:
        torch.manual_seed(seed)
        idx = torch.randint(0, len(dataset), (1,)).item()
        dat = dataset[idx]
        for k, v in dat.items():
            if k not in data:
                data[k] = []
            data[k].append(v)
    for k, v in data.items():
        data[k] = torch.stack(v, dim=0)
        data[k] = data[k].to(device)
    return data


def save_scene(path, meshes, colors):
    # Compresse colors.
    for i in range(len(colors)):
        colors[i] = np.clip(colors[i] * 255, 0, 255).astype(np.uint8)
        with io.BytesIO() as f:
            imageio.imwrite(f, colors[i], format='png')
            colors[i] = f.getvalue()
    depths = []
    for i in range(len(meshes)):
        image_size = meshes[i].depth.shape[0]
        depth = np.ascontiguousarray(meshes[i].depth.astype(np.float32))
        depth = np.frombuffer(depth, dtype=np.uint8).reshape(image_size, image_size, 4)
        with io.BytesIO() as f:
            imageio.imwrite(f, depth, format='png')
            depths.append(f.getvalue())

    data = [
        {
            'color': colors[i],
            'depth': depths[i],
            'fov': meshes[i].fov,
            'modelview': meshes[i].modelview,
        }
        for i in range(len(meshes))
    ]

    # npz
    np.savez_compressed(path, data=data)


def load_scene(path, atol=0.03, rtol=0.03, erode_rgb=3):
    # npz
    data = np.load(path, allow_pickle=True)['data']
    image_size = imageio.imread(io.BytesIO(data[0]['color'])).shape[0]
    meshes = [rgbd_3d.utils.depth_to_mesh(
        np.frombuffer(imageio.imread(io.BytesIO(d['depth'])), dtype=np.float32).reshape(image_size, image_size, 1),
        32, d['fov'], d['modelview'], atol=atol, rtol=rtol, erode_rgb=erode_rgb, cal_normal=True
    ) for d in data]
    colors = [imageio.imread(io.BytesIO(data[i]['color'])) / 255 for i in range(len(data))]
    return meshes, colors


def load_first_view(path, near=0.6, far=5):
    # npz
    data = np.load(path, allow_pickle=True)['data'][0]
    color = imageio.imread(io.BytesIO(data['color'])) / 255
    depth = np.frombuffer(imageio.imread(io.BytesIO(data['depth'])), dtype=np.float32).reshape(color.shape[0], color.shape[0], 1)
    depth = rgbd_3d.utils.project_depth(depth, near, far)
    return np.concatenate([color, depth], axis=-1)
