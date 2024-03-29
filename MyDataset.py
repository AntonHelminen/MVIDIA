import os
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image


class MyDataset(Dataset):

    def __init__(self, path, transform, target_transform):
        self.root = path        
        self.images = os.listdir(self.root)
        self.transforms = transform
        self.target_transform = target_transform

    
    def __len__(self):
        return len(self.images)


    def __getitem__(self, idx):
        file = self.images[idx]
        label = int(file.split('_')[0]) - 1
        image = read_image(f'{self.root}/{file}')
        image = self.transforms(image)
        label = self.target_transform(label)
        return image, label


    def get_image(self, idx):
        file = self.images[idx]
        label = int(file.split('_')[0]) - 1
        image = read_image(f'{self.root}/{file}')
        return image, label
