import json
import os
import glob

from .base import BaseDataset, SRDataset, WarpDataset


class ImageNet(BaseDataset):
    """
    ImageNet dataset

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
        super().__init__(root_path, image_size, normalize, normalize_depth, prepocess_depth, near, far)

    def get_fileinfo(self):
        if os.path.isfile(os.path.join(self.root_path, 'dataset.json')):
            info = json.load(open(os.path.join(self.root_path, 'dataset.json'), 'r'))
            self.labels = info['labels']
            self.images = info['images']
            self.depths = info['depths']
        else:
            self.labels = os.listdir(os.path.join(self.root_path, 'images'))
            assert len(self.labels) > 0, "Can't find data; make sure you specify the path to your dataset"
            self.labels.sort()
            self.images = []
            self.depths = []
            for label in self.labels:
                tmp = [os.path.relpath(i, self.root_path) for i in glob.glob(os.path.join(self.root_path, 'images', label, '*.JPEG'))]
                self.images += tmp
                self.depths += [os.path.join('depths', label, f.split('/')[-1].replace('JPEG','npz')) for f in tmp]
            self.labels = {c: i for i, c in enumerate(self.labels)}
            json.dump({
                'labels': self.labels,
                'images': self.images,
                'depths': self.depths
            }, open(os.path.join(self.root_path, 'dataset.json'), 'w'))


class ImageNetSR(SRDataset, ImageNet):
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
        super().__init__(root_path, image_size, image_size_lr, normalize, normalize_depth, prepocess_depth, near, far)


class ImageNetWarp(WarpDataset, ImageNet):
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
        super().__init__(root_path, image_size, normalize, normalize_depth, prepocess_depth, near, far, augments, std)
