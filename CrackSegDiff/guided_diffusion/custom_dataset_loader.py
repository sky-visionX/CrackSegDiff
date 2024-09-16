import os
from torch.utils.data import Dataset
from PIL import Image
from glob import glob
import tifffile as tiff
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import squeezenet1_1
class CustomDataset(Dataset):
    def __init__(self, args, data_path, transform=None, mode='Training'):

        print("loading data from the directory :",data_path)
        path = data_path
        images = sorted(glob(os.path.join(path, "5d/*.png")))
        masks = sorted(glob(os.path.join(path, "mask/*.bmp")))

        self.name_list = images
        self.label_list = masks
        self.data_path = path
        self.mode = mode

        self.transform = transform

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        """Get the images"""
        name = self.name_list[index]
        img_path = os.path.join(name)
        
        mask_name = self.label_list[index]
        msk_path = os.path.join(mask_name)
        img = tiff.imread(img_path)
        # img = Image.open(img_path)
        # img = Image.open(img_path).convert('RGB')
        mask = Image.open(msk_path).convert('L')
        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            torch.set_rng_state(state)
            mask = self.transform(mask)

        if self.mode == 'Training':
            return (img, mask, name)
        else:
            return (img, mask, name)