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

# class SpineSet(data.Dataset):
#     def __init__(self,root):
#         fh = open(root, 'r', encoding='utf-8')
#         imgs = []
#         for line in fh:
#             words = line.split(',')
#         label = []
#         for i in range(len(words)-1):
#             if len(words[i].split('\''))>1:
#                 path = words[i].split('\'')[1]
#                 imgs.append(path)
#         np.random.seed(200)
#         random.shuffle(imgs)
#         for path in imgs:
#             if path.split('data\\\\data\\\\')[1].split(' ')[0] == 'sch':
#                 label.append(1)
#             elif path.split('data\\\\data\\\\')[1].split(' ')[0] == 'ep':
#                 label.append(0)
#
#         self.labels=label
#         self.imgs=imgs
#         self.transforms=transform
#
#     def __getitem__(self,index):
#
#         img_path=self.imgs[index]
#         pil_img=Image.open(img_path)
#         if self.transforms:
#             data=self.transforms(pil_img)
#         else:
#             pil_img=np.asarray(pil_img)
#             data=torch.from_numpy(pil_img)
#         self.x_data=data
#         self.y_data=self.labels[index]
#         return self.x_data,self.y_data
#
#     def __path__(self,index):
#         return self.imgs[index]
#
#     def __len__(self):
#         return len(self.imgs)
#
# class ValidSet(data.Dataset):
#     def __init__(self,root):
#         fh = open(root, 'r', encoding='utf-8')
#         imgs = []
#         for line in fh:
#             words = line.split(',')
#         label = []
#         for i in range(len(words)-1):
#             if len(words[i].split('\''))>1:
#                 path = words[i].split('\'')[1]
#                 imgs.append(path)
#         for path in imgs:
#             if path.split('data\\\\data\\\\')[1].split(' ')[0] == 'sch':
#                 label.append(1)
#             elif path.split('data\\\\data\\\\')[1].split(' ')[0] == 'ep':
#                 label.append(0)
#         self.labels=label
#         self.imgs=imgs
#         self.transforms = validtransform
#
#     def __getitem__(self,index):
#         img_path=self.imgs[index]
#         pil_img=Image.open(img_path)
#         if self.transforms:
#             data = self.transforms(pil_img)
#         else:
#             pil_img = np.asarray(pil_img)
#             data = torch.from_numpy(pil_img)
#         self.x_data=data
#         self.y_data=self.labels[index]
#         return self.x_data,self.y_data
#
#     def __len__(self):
#         return len(self.imgs)

# root='E:\\dwl\\gzw\\2dclassify\\test 10 cut'
# imgs=[]
# type=['T2']
# for dirname,_,filelist in os.walk(root):
#     for filename in filelist:
#         print(os.path.join(dirname,filename))
#         if 'T2' in os.path.join(dirname,filename).split('\\')[-2]:
#             imgs.append(os.path.join(dirname,filename))
# #print(imgs)
#
# path='E:\\dwl\\gzw\\2dclassify\\'+type[0]+'_in_test.txt'
# # fh = open(path, 'w')
# # for i in range(len(imgs)):
# #     s=str(imgs[i]).replace('[','').replace(']','')
# #     s=s.replace("'",'')+','+'\n'
# #     fh.write(s)
# # fh.close()


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
        fh = open(root, 'r', encoding='utf-8')
        imgs = []
        for line in fh:
            imgs.append(line.split(',')[0])
        np.random.seed(200)
        random.shuffle(imgs)
        label = []
        # print(imgs)
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
        fh = open(root, 'r', encoding='utf-8')
        imgs = []
        for line in fh:
            imgs.append(line.split(',')[0])

        label = []
        # print(imgs)
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
        fh = open(root, 'r', encoding='utf-8')
        imgs = []
        for line in fh:
            imgs.append(line.split(',')[0])

        label = []
        #print(imgs)
        for path in imgs:
            if 'sch' in path.split('\\')[-2]:
                label.append(1)
            elif 'ep' in path.split('\\')[-2]:
                label.append(0)
        #print(label)
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
# class SpineSet(data.Dataset):
#     def __init__(self,root):
#         fh = open(root, 'r', encoding='utf-8')
#         imgs = []
#         for line in fh:
#             words = line.split(',')
#         label = []
#         for i in range(len(words)-1):
#             if len(words[i].split('\''))>1:
#                 path = words[i].split('\'')[1]
#                 imgs.append(path)
#         np.random.seed(200)
#         random.shuffle(imgs)
#         for path in imgs:
#             if path.split('data\\\\data\\\\')[1].split(' ')[0] == 'sch':
#                 label.append(1)
#             elif path.split('data\\\\data\\\\')[1].split(' ')[0] == 'ep':
#                 label.append(0)
#
#         self.labels=label
#         self.imgs=imgs
#         self.transforms=transform
#
#     def __getitem__(self,index):
#
#         img_path=self.imgs[index]
#         pil_img=Image.open(img_path)
#         if self.transforms:
#             data=self.transforms(pil_img)
#         else:
#             pil_img=np.asarray(pil_img)
#             data=torch.from_numpy(pil_img)
#         self.x_data=data
#         self.y_data=self.labels[index]
#         return self.x_data,self.y_data,self.imgs[index]
#
#     def __path__(self,index):
#         return self.imgs[index]
#
#     def __len__(self):
#         return len(self.imgs)
#
# class ValidSet(data.Dataset):
#     def __init__(self,root):
#         fh = open(root, 'r', encoding='utf-8')
#         imgs = []
#         for line in fh:
#             words = line.split(',')
#         label = []
#         for i in range(len(words)-1):
#             if len(words[i].split('\''))>1:
#                 path = words[i].split('\'')[1]
#                 imgs.append(path)
#         for path in imgs:
#             if path.split('data\\\\data\\\\')[1].split(' ')[0] == 'sch':
#                 label.append(1)
#             elif path.split('data\\\\data\\\\')[1].split(' ')[0] == 'ep':
#                 label.append(0)
#         self.labels=label
#         self.imgs=imgs
#         self.transforms = validtransform
#
#     def __getitem__(self,index):
#         img_path=self.imgs[index]
#         pil_img=Image.open(img_path)
#         if self.transforms:
#             data = self.transforms(pil_img)
#         else:
#             pil_img = np.asarray(pil_img)
#             data = torch.from_numpy(pil_img)
#         self.x_data=data
#         self.y_data=self.labels[index]
#         return self.x_data,self.y_data,self.imgs[index]
#
#     def __len__(self):
#         return len(self.imgs)
#
# class TestSet(data.Dataset):
#     def __init__(self,root):
#         fh = open(root, 'r', encoding='utf-8')
#         imgs = []
#         for line in fh:
#             imgs.append(line.split(',')[0])
#
#         label = []
#         #print(imgs)
#         for path in imgs:
#             if 'sch' in path.split('va-')[1].split('0')[0] :
#                 label.append(1)
#             elif 'ep' in path.split('va-')[1].split('0')[0] :
#                 label.append(0)
#         #print(label)
#         self.labels = label
#         self.imgs = imgs
#         self.transforms = validtransform
#
#     def __getitem__(self,index):
#         img_path=self.imgs[index]
#         pil_img=Image.open(img_path)
#         if self.transforms:
#             data = self.transforms(pil_img)
#         else:
#             pil_img = np.asarray(pil_img)
#             data = torch.from_numpy(pil_img)
#         self.x_data=data
#         self.y_data=self.labels[index]
#         return self.x_data,self.y_data,self.imgs[index]
#
#     def __len__(self):
#         return len(self.imgs)
# type='T2'
# i=1
# root='E:\\dwl\\gzw\\2dclassify\\' + type + '\\train_%s.txt' % (i + 1)
#
# fh = open(root, 'r', encoding='utf-8')
# imgs = []
# for line in fh:
#     imgs.append(line.split(',')[0])
#
# label = []
# print(len(imgs),imgs)
# for path in imgs:
#     if 'sch' in path.split('\\')[-2]:
#         label.append(1)
#     elif 'ep' in path.split('\\')[-2]:
#         label.append(0)
# print(len(label),label)

