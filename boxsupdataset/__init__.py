import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import glob
import pandas as pd
from skimage import io, transform


class NasaBoxSupDataset(Dataset):
    """ Nasa Box Sup dataset. """

    def __init__(self, classfile, rootDir, transform=None):
        """
        Args:
            root_dir (string): Directory with img folder and label folder.
            transform (callable, optional): Optional transform to be applied.
        """
        assert (Path(rootDir) / 'Images').exists and \
            (Path(rootDir) / 'Labels').exists, \
            'rootDir has not the Correct Format'
        assert callable(transform) or transform is None, \
            'transform needs to be a callable'

        self.__rootDir = Path(rootDir)
        self.__transform = transform
        self.__imgDir = self.rootDir / 'Images'
        self.__labelDir = self.rootDir / 'Labels'
        self.__classes = pd.read_csv(self.__labelDir / classfile)

    def __len__(self):
        return len(glob.glob1(self.imgDir, "*.png"))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.toList()

        img_files = glob.glob1(self.imgDir, "*.png")
        img_name = img_files[idx]
        label_files = glob.glob1(self.labelDir, "*.png")
        label_name = label_files[idx]
        image = io.imread(self.imgDir / img_name)
        label = io.imread(self.labelDir / label_name)
        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample

    ###########################################################################
    #                       G E T T E R  &  S E T T E R                       #
    ###########################################################################
    # #--> Getter:
    def __getRootDir(self):
        return self.__rootDir

    def __getTransform(self):
        return self.__transform

    def __getImgDir(self):
        return self.__imgDir

    def __getLabelDir(self):
        return self.__labelDir

    # #--> Setter:
    def __setRootDir(self, x):
        if isinstance(x, Path):
            self.__rootDir = x
        else:
            raise ValueError(x, 'rootDir needs to be Path instance.')

    def __setTransform(self, x):
        if callable(x):
            self.__transform = x
        else:
            raise ValueError(x, 'transform need to be a callable.')

    def __setImgDir(self, x):
        if isinstance(x, Path):
            self.__imgDir = x
        else:
            raise ValueError(x, 'ImgDir needs to be Path instance.')

    def __setLabelDir(self, x):
        if isinstance(x, Path):
            self.__labelDir = x
        else:
            raise ValueError(x, 'LabelDir needs to be Path instance.')

    # #--> Properties:
    rootDir = property(__getRootDir, __setRootDir)
    transform = property(__getTransform, __setTransform)
    imgDir = property(__getImgDir, __setImgDir)
    labelDir = property(__getLabelDir, __setLabelDir)


# FOR DEBUGGING #
if __name__ == "__main__":
    testDataset = NasaBoxSupDataset('classes_bxsp.txt', 'data/TestBatch')
    print(len(testDataset))
    print(testDataset[3])
