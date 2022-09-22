import os
import torch
import torch.utils.data
from PIL import Image
import pandas as pd

## TODO: Need to refactor this class to work wit ISIC and other datasets
# Input ->       Path to folder.
# Output -> x   (nd.Array, label) for Training
#                (nd.Array) for Testing
# This is for the ISIC dataset format. The implementation is very specific to the way the ISIC
# dataset has been presented


class ISICDataset(torch.utils.data.Dataset):
    """
    root - path to the folder containing the images
    img_list - path to the csv containing the images and their ground truth labels
    transforms - transforms to be done on the dataset
    """

    def __init__(self, root, img_list, transforms=None):
        self.root = root
        self.transforms = transforms
        self.img_list = pd.read_csv(img_list)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        raise NotImplementedError


class ISICTrainDataset(ISICDataset):
    def __init__(self, root, img_list, transforms=None):
        super().__init__(root, img_list, transforms)

    def __getitem__(self, idx):
        img_path = self.root + "/" + self.img_list["image"][idx] + ".jpg"
        img = Image.open(img_path).convert("RGB")
        label = self.img_list.to_numpy().nonzero()[0][1]

        if self.transforms is not None:
            img = self.transforms(img)

        return img, label


class ISICTestDataset(ISICDataset):
    def __init__(self, root, img_list, transforms=None):
        super().__init__(root, img_list, transforms)

    def __getitem__(self, idx):
        img_path = self.root + "/" + self.img_list["image"][idx] + ".jpg"
        # print(img_path)
        img = Image.open(img_path).convert("RGB")

        if self.transforms is not None:
            img = self.transforms(img)

        return img
