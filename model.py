import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.utils import data
from torch.utils.data import DataLoader
import torch.optim as optim
from efficientnet_pytorch import EfficientNet
import torchvision
from torchvision import transforms
from dataset import SpineSet
from PIL import Image
import numpy as np
import random
import math
import argparse

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet,self).__init__()
        #self.conv=nn.Conv2d(1,3,kernel_size=1)
        self.resnet=torchvision.models.resnet50(pretrained=True)
        self.fc=torch.nn.Linear(1000,2)

    def forward(self,x):
        x=self.resnet(x)
        x = F.relu(x)
        x=self.fc(x)
        x = torch.sigmoid(x)
        return x

class VGG(nn.Module):
    def __init__(self):
        super(VGG,self).__init__()
        self.resnet=torchvision.models.vgg19(pretrained=True)
        self.fc = torch.nn.Linear(1000, 2)

    def forward(self,x):
        x=self.resnet(x)
        x = F.relu(x)
        x=self.fc(x)
        x = torch.sigmoid(x)
        return x

class Inception(nn.Module):
    def __init__(self):
        super(Inception,self).__init__()
        self.conv=nn.Conv2d(3,3,kernel_size=1)
        self.inception=torchvision.models.inception_v3(aux_logits=False,pretrained=True)
        self.fc=torch.nn.Linear(1000,2)

    def forward(self,x):
        x=self.inception(x)
        x = F.relu(x)
        x=self.fc(x)
        x = torch.sigmoid(x)
        return x

class Efficientnet(nn.Module):
    def __init__(self):
        super(Efficientnet, self).__init__()
        self.conv = nn.Conv2d(3, 3, kernel_size=1)
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b2')
        self.fc = torch.nn.Linear(1000, 2)

    def forward(self, x):
        # x=self.conv(x)
        x = self.efficientnet(x)
        x = F.relu(x)
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x
