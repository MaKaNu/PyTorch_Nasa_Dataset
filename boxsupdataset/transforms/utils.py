""" This utils module include diverse transfrom classes.
"""

from __future__ import absolute_import
import torch
import numpy as np
from skimage.util import img_as_float64


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample: dict) -> dict:
        image, label = sample['image'], sample['label']

        if len(image.shape) == 2:
            image = np.dstack([image]*3)

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        label = label.transpose((2, 0, 1))
        return {'image': torch.from_numpy(img_as_float64(image)),
                'label': torch.from_numpy(img_as_float64(label))}
