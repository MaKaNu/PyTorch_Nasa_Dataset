""" This module holds the single class NasaBoxSupDataset. This Dataset loads
    images together with the labeldata as png. Also transformations on the
    images are computed.
"""

from __future__ import absolute_import
from pathlib import Path
import glob
import torch
from torch.utils.data import Dataset
import pandas as pd
from skimage import io


class NasaBoxSupDataset(Dataset):
    """ Nasa Box Sup dataset. """

    def __init__(self, classfile, rootDir, transform=None):
        """
        Args:
            root_dir (string): Directory with img folder and label folder.
            transform (callable, optional): Optional transform to be applied.
        """
        assert (Path(rootDir) / 'Images').exists() and \
            (Path(rootDir) / 'Labels').exists(), \
            'rootDir has not the Correct Format or does not exists.'
        assert callable(transform) or transform is None, \
            'transform needs to be a callable.'

        self.__root_dir = Path(rootDir)
        self.__transform = transform
        self.__img_dir = self.root_dir / Path('Images')
        self.__label_dir = self.root_dir / Path('Labels')
        self.__classes = pd.read_csv(self.__label_dir / Path(classfile))

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

        if self.transform:
            sample = self.transform(sample)

        return sample

    ###########################################################################
    #                       G E T T E R  &  S E T T E R                       #
    ###########################################################################
    # #--> Getter:
    def __getRootDir__(self):
        return self.__root_dir

    def __getTransform__(self):
        return self.__transform

    def __getImgDir__(self):
        return self.__img_dir

    def __getLabelDir__(self):
        return self.__label_dir

    def __getClasses__(self):
        return self.__classes

    # #--> Setter:
    def __setRootDir__(self, val_to_set):
        if isinstance(val_to_set, Path):
            self.__root_dir = val_to_set
        else:
            raise ValueError(val_to_set, 'root_dir needs to be Path instance.')

    def __setTransform__(self, val_to_set):
        if callable(val_to_set):
            self.__transform = val_to_set
        else:
            raise ValueError(val_to_set, 'transform need to be a callable.')

    def __setImgDir__(self, val_to_set):
        if isinstance(val_to_set, Path):
            self.__img_dir = val_to_set
        else:
            raise ValueError(val_to_set, 'img_dir needs to be Path instance.')

    def __setLabelDir__(self, val_to_set):
        if isinstance(val_to_set, Path):
            self.__label_dir = val_to_set
        else:
            raise ValueError(val_to_set, 'label_dir needs to be Path instance.')

    def __setClasses__(self, val_to_set):
        if isinstance(val_to_set, pd.core.frame.DataFrame):
            self.__classes = val_to_set
        else:
            raise ValueError(val_to_set, 'classes needs to be Path instance.')

    # #--> Properties:
    root_dir = property(__getRootDir__, __setRootDir__)
    transform = property(__getTransform__, __setTransform__)
    img_dir = property(__getImgDir__, __setImgDir__)
    label_dir = property(__getLabelDir__, __setLabelDir__)
    classes = property(__getClasses__, __setClasses__)
