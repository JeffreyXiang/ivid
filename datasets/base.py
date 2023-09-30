from functools import partial
import os

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import PIL
import numpy as np
import cv2
import time

import glm
import rgbd_3d


class BaseDataset(Dataset):
    """
    BaseDataset.
    Load RGBD images and labels (if available)

    Args:
        root_path (str): path to the dataset
        image_size (int): size of the images
        normalize (bool): whether to normalize the images to [-1, 1]
        normalize_depth (bool): whether to normalize the depth maps to [-1, 1]
        prepocess_depth (str): how to preprocess the depth maps (inputs from the dataset are disparity maps)
            - 'none': no preprocessing
            - 'to_depth': disparity map, to depth map
            - 'disparity_minmax': disparity map, min-max normalization, min=0, max=1
            - 'depth_minmax': depth map, min-max normalization, min=0, max=1
            - 'z_buffer': perspective projection to [0, 1]
        near (float): near plane for perspective projection
        far (float): far plane for perspective projection
    """
    def __init__(self,
        root_path,
        image_size,
        normalize=False,
        normalize_depth=False,
        prepocess_depth='none',
        near=0.5,
        far=100,
    ):
        super().__init__()

        assert prepocess_depth in ['none', 'to_depth', 'disparity_minmax', 'depth_minmax', 'z_buffer'], "Unknown depth preprocessing method"
        assert not (normalize_depth and (prepocess_depth == 'none' or prepocess_depth == 'to_depth')), "Can't normalize depth maps if they are not mapped to [0, 1]"

        self.root_path = root_path
        self.image_size = image_size
        self.normalize = normalize
        self.normalize_depth = normalize_depth
        self.prepocess_depth = prepocess_depth
        self.near = near
        self.far = far

        self.images = None
        self.depths = None
        self.labels = None
        
        self.get_fileinfo() # set self.images, self.depths, self.labels

        self.num_classes = len(self.labels) if self.labels is not None else None

        self.transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=F.InterpolationMode.LANCZOS),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ])
        self.transform_depth = transforms.Compose([
            transforms.Resize(image_size, interpolation=F.InterpolationMode.NEAREST),
            transforms.CenterCrop(image_size),
        ])

    def to3channel(self, image):
        if image.shape[0] == 1: image = image.repeat(3, 1, 1)
        if image.shape[0] == 4: image = image[:3]
        return image

    def get_fileinfo(self):
        """
        Set labels, images, and depths.
        This function is called when the dataset is initialized.
        Should be implemented in the child class.
        """
        pass

    def get_file(self, index):
        image = PIL.Image.open(os.path.join(self.root_path, self.images[index]))

        depth = np.load(os.path.join(self.root_path, self.depths[index]))['arr_0'].astype(np.float32)
        depth /= 6250
        if depth.max() > 1 / self.near:
            depth /= depth.max() * self.near
        depth = np.maximum(depth, 1e-3)

        if self.prepocess_depth == 'none':
            pass
        elif self.prepocess_depth == 'to_depth':
            depth = 1 / depth
        elif self.prepocess_depth == 'disparity_minmax':
            depth = (depth - depth.min()) / (depth.max() - depth.min())
        elif self.prepocess_depth == 'depth_minmax':
            depth = 1 / depth
            depth = (depth - depth.min()) / (depth.max() - depth.min())
        elif self.prepocess_depth == 'z_buffer':
            depth = (depth - 1 / self.near) / (1 / self.far - 1 / self.near)
            depth = np.clip(depth, 0, 1)

        depth = PIL.Image.fromarray(depth)

        label = self.labels[self.images[index].split('/')[-2]] if self.num_classes is not None else None

        return image, depth, label
    
    def process_file(self, image, depth, label):
        image = self.transform(image)
        image = self.to3channel(image)
        if self.normalize:
            image = image * 2 - 1

        depth = self.transform_depth(depth)
        depth = transforms.ToTensor()(np.array(depth).astype(np.float32))
        if self.normalize_depth:
            depth = depth * 2 - 1

        data = {
            'x_0': torch.cat([image, depth])
        }

        if label is not None:
            data['classes'] = torch.tensor(label)

        return data

    def getitem(self, index):
        image, depth, label = self.get_file(index)
        return self.process_file(image, depth, label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        try:
            return self.getitem(index)
        except Exception as e:
            print(e)
            return self.__getitem__(np.random.randint(self.__len__()))


class SRDataset(BaseDataset):
    """
    SRDataset.
    Load RGBD images and labels (if available) for super-resolution
    """
    def __init__(self,
        root_path,
        image_size,
        image_size_lr,
        normalize=False,
        normalize_depth=False,
        prepocess_depth='none',
        near=0.5,
        far=100,
    ):
        super().__init__(root_path, image_size, normalize, normalize_depth, prepocess_depth, near, far)

        self.transform_lr = transforms.Compose([
            transforms.Resize(image_size_lr, interpolation=F.InterpolationMode.LANCZOS),
            transforms.CenterCrop(image_size_lr),
        ])
        self.transform_depth_lr = transforms.Compose([
            transforms.Resize(image_size_lr, interpolation=F.InterpolationMode.NEAREST),
            transforms.CenterCrop(image_size_lr),
        ])

    def process_file(self, image, depth, label):
        data = super().process_file(image, depth, label)
        image_lr = np.array(self.transform_lr(image))
        image_lr = cv2.GaussianBlur(image_lr, (3, 3), np.random.rand() + 1e-3)
        image_lr = transforms.ToTensor()(image_lr)
        image_lr = self.to3channel(image_lr)
        if self.normalize:
            image_lr = image_lr * 2 - 1

        depth_lr = self.transform_depth_lr(depth)
        depth_lr = transforms.ToTensor()(np.array(depth_lr).astype(np.float32))
        if self.normalize_depth:
            depth_lr = depth_lr * 2 - 1

        data['y'] = torch.cat([image_lr, depth_lr])

        return data


class WarpDataset(BaseDataset):
    def __init__(
        self,
        root_path,
        image_size,
        normalize=False,
        normalize_depth=False,
        prepocess_depth='none',
        near=0.5,
        far=100,
        augments=[],
        std=0.15,
    ):
        super().__init__(root_path, image_size, normalize, normalize_depth, prepocess_depth, near, far)
        self.renderer = None
        self.augments = augments
        self.std = std

    def __getitem__(self, index):
        data = super().__getitem__(index)
        if self.renderer is None:
            device_id = torch.cuda.current_device()
            self.renderer = rgbd_3d.SimpleRenderer(self.image_size * 3, self.image_size, near=0.1, far=200, device=device_id)
       
        rgbd = data['x_0'].cpu().permute(1, 2, 0).numpy().copy()
        if self.normalize: rgbd[..., :3] = rgbd[..., :3] * 0.5 + 0.5
        if self.normalize_depth: rgbd[..., 3:] = rgbd[..., 3:] * 0.5 + 0.5
        x_0 = rgbd.copy()

        if 'prewarp_noise' in self.augments:
            rgbd = rgbd + np.random.normal(0, 0.005 * np.random.rand(), rgbd.shape)
        
        # warp
        theta = np.random.randn() * self.std
        phi = np.random.randn() * self.std
        r = 1 + np.random.randn() * 0.1
        modelview = glm.lookAt(
            glm.vec3(r * np.cos(phi) * np.sin(theta), r * np.sin(phi), r * np.cos(phi) * np.cos(theta)),
            glm.vec3(np.random.randn((3)) * 0.05),
            glm.vec3(0.0, 1.0, 0.0)
        )
        res = rgbd_3d.utils.forward_backward_warp(self.renderer, rgbd, modelview, near=self.near, far=self.far, padding=self.image_size)
        y = np.concatenate([res.color, res.depth], axis=-1)
        mask = res.mask
 
        if 'postwarp_noise' in self.augments:
            y = y + np.random.normal(0, 0.03 * np.random.rand(), y.shape)

        if 'blur' in self.augments:
            if np.random.rand() < 0.8:
                y[:, :, :3] = cv2.GaussianBlur(x_0[:, :, :3], (3, 3), np.random.rand() + 1e-3)

        if 'erode_rgb' in self.augments:
            erode_radius = 2 * np.random.randint(5) + 1
            if erode_radius > 0:
                mask_rgb = cv2.erode(mask[..., 0], np.ones((erode_radius, erode_radius)))
                y[:, :, :3] *= mask_rgb[..., None]
                mask_rgb = torch.from_numpy(mask_rgb)[None]
                data['mask_rgb'] = mask_rgb.float()

        y = torch.from_numpy(y).permute(2, 0, 1)
        mask = torch.from_numpy(mask).permute(2, 0, 1)
        y *= mask
        if self.normalize: y[..., :3] = y[..., :3] * 2 - 1
        if self.normalize_depth: y[..., 3:] = y[..., 3:] * 2 - 1

        data['y'] = y.float()
        data['mask'] = mask.float()
        data['pose'] = torch.tensor([theta, phi]).float()

        return data