import torchvision
import os
import torch
from torch.utils import data
from PIL import Image
import numpy as np
from torchvision import transforms
import torch
import torch.nn as nn
import random
import math
import torch.utils.model_zoo as model_zoo
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

transform=transforms.Compose([
    transforms.RandomRotation(degrees=40),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=0,translate=(0.2,0.2),scale=(1.5,1.5),shear=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5,.5,.5],std=[.5,.5,.5]),
    transforms.Resize([500,500])
])
validtransform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5,.5,.5],std=[.5,.5,.5]),
    transforms.Resize([500,500])
])

class SpineSet(data.Dataset):
    def __init__(self,root):
        # root <str>: path to .txt file consisted of training img path
        
        fh = open(root, 'r', encoding='utf-8')
        imgs = []
        for line in fh:
            imgs.append(line.split(',')[0])
        np.random.seed(200)
        random.shuffle(imgs)
        label = []
        for path in imgs:
            if 'sch' in path.split('\\')[-2]:
                label.append(1)
            elif 'ep' in path.split('\\')[-2]:
                label.append(0)

        self.labels=label
        self.imgs=imgs
        self.transforms=transform

    def __getitem__(self,index):
        img_path=self.imgs[index]
        pil_img=Image.open(img_path)
        if self.transforms:
            data=self.transforms(pil_img)
        else:
            pil_img=np.asarray(pil_img)
            data=torch.from_numpy(pil_img)
        self.x_data=data
        self.y_data=self.labels[index]
        return self.x_data,self.y_data

    def __path__(self,index):
        return self.imgs[index]

    def __len__(self):
        return len(self.imgs)

class ValidSet(data.Dataset):
    def __init__(self,root):
        # root <str>: path to .txt file consisted of validation img path
        # return img path
        
        fh = open(root, 'r', encoding='utf-8')
        imgs = []
        for line in fh:
            imgs.append(line.split(',')[0])

        label = []
        for path in imgs:
            if 'sch' in path.split('\\')[-2]:
                label.append(1)
            elif 'ep' in path.split('\\')[-2]:
                label.append(0)

        self.labels=label
        self.imgs=imgs
        self.transforms = validtransform

    def __getitem__(self,index):
        img_path=self.imgs[index]
        pil_img=Image.open(img_path)
        if self.transforms:
            data = self.transforms(pil_img)
        else:
            pil_img = np.asarray(pil_img)
            data = torch.from_numpy(pil_img)
        self.x_data=data
        self.y_data=self.labels[index]
        return self.x_data,self.y_data,self.imgs[index]

    def __len__(self):
        return len(self.imgs)

class TestSet(data.Dataset):
    def __init__(self,root):
        # root <str>: path to .txt file consisted of test img path
        # return img path
        
        fh = open(root, 'r', encoding='utf-8')
        imgs = []
        for line in fh:
            imgs.append(line.split(',')[0])

        label = []
        for path in imgs:
            if 'sch' in path.split('\\')[-2]:
                label.append(1)
            elif 'ep' in path.split('\\')[-2]:
                label.append(0)
        
        self.labels = label
        self.imgs = imgs
        self.transforms = validtransform

    def __getitem__(self,index):
        img_path=self.imgs[index]
        pil_img=Image.open(img_path)
        if self.transforms:
            data = self.transforms(pil_img)
        else:
            pil_img = np.asarray(pil_img)
            data = torch.from_numpy(pil_img)
        self.x_data=data
        self.y_data=self.labels[index]
        return self.x_data,self.y_data,self.imgs[index]

    def __len__(self):
        return len(self.imgs)
