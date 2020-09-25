""" This module holds the single class NasaBoxSupDataset. This Dataset loads
    images together with the labeldata as png. Also transformations on the
    images are computed.
"""

from __future__ import absolute_import
from pathlib import Path
import glob
import torch
from torch.utils.data import Dataset
from pandas.core.frame import DataFrame
import pandas as pd
from skimage import io


class NasaBoxSupDataset(Dataset):
    """ Nasa Box Sup dataset. """

    def __init__(
        self, classfile, root_dir, transform=None, target_transfrom=None):
        """
        Args:
            root_dir (string): Directory with img folder and label folder.
            transform (callable, optional): Optional transform to be applied.
        """
        assert (Path(root_dir) / 'Images').exists() and \
            (Path(root_dir) / 'Labels').exists(), \
            'rootDir has not the Correct Format or does not exists.'
        assert callable(transform), \
            'transform needs to be a callable.'

        self.root_dir = Path(root_dir)
        self.transform = transform
        self.target_transform = target_transfrom
        self.img_dir = self.root_dir / Path('Images')
        self.label_dir = self.root_dir / Path('Labels')
        self.classes = pd.read_csv(self.label_dir / Path(classfile))

    def __len__(self):
        return len(glob.glob1(self.img_dir, "*.png"))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.toList()

        img_files = glob.glob1(self.img_dir, "*.png")
        img_name = img_files[idx]
        label_files = glob.glob1(self.label_dir, "*.png")
        label_name = label_files[idx]
        image = io.imread(self.img_dir / Path(img_name))
        label = io.imread(self.label_dir / Path(label_name))
        sample = {'image': image, 'label': label}

        if self.transform is not None:
            sample['image'] = self.transform(sample['image'])
        if self.target_transform is not None:
            sample['label'] = self.transform(sample['label'])

        return sample

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}'
                f'Attirbutes:'
                f'root_dir={self.root_dir},'
                f'transfomrs={self.transform},'
                f'img_dir={self.img_dir},'
                f'label_dir={self.label_dir},'
                f'classes={self.classes}'
                )

    def make_dataset(self):
        pass # TODO
    
    @property
    def root_dir(self):
        """ root_dir Getter"""
        return self._root_dir
    
    @root_dir.setter
    def root_dir(self, value):
        if not isinstance(value, Path):
            raise TypeError("value needs to be of Type Path")
        self._root_dir = value

    @property
    def transform(self):
        """ transform Getter"""
        return self._transform
    
    @transform.setter
    def transform(self, value):
        if not (callable(value) or value is None):
            raise TypeError("value needs to be a callable")
        self._transform = value

    @property
    def target_transform(self):
        """ target_transform Getter"""
        return self._target_transform
    
    @target_transform.setter
    def target_transform(self, value):
        if not (callable(value) or value is None):
            raise TypeError("value needs to be a callable")
        self._target_transform = value

    @property
    def img_dir(self):
        """ img_dir Getter"""
        return self._img_dir
    
    @img_dir.setter
    def img_dir(self, value):
        if not isinstance(value, Path):
            raise TypeError("value needs to be of Type Path")
        self._img_dir = value

    @property
    def label_dir(self):
        """ label_dir Getter"""
        return self._label_dir
    
    @label_dir.setter
    def label_dir(self, value):
        if not isinstance(value, Path):
            raise TypeError("value needs to be of Type Path")
        self._label_dir = value

    @property
    def classes(self):
        """ classes Getter"""
        return self._classes
    
    @classes.setter
    def classes(self, value):
        if not isinstance(value, DataFrame):
            raise TypeError("value needs to be of Type DataFrame")
        self._classes = value
    