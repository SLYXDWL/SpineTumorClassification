import os
#os.environ['CUDA_VISIBLE_DEVICES']='4,5,6'

import torchvision
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
from dataset import SpineSet
from dataset import ValidSet
from model import ResNet
from model import Inception
import cv2

from torch.utils.data import DataLoader
parser=argparse.ArgumentParser(description='PyTorch Example')
parser.add_argument('--batch-size',type=int,default=64,metavar='N',help='input batch size for training(default:64)')
parser.add_argument('--test-batch-size',type=int,default=10,metavar='N',help='input batch size for testing(default:10)')
parser.add_argument('--epochs',type=int,default=10,metavar='N',help='number of epochs to train(default:10)')
parser.add_argument('--lr',type=float,default=0.01,metavar='LR',help='learning rate(default:0.01)')
parser.add_argument('--momentum',type=float,default=0.5,metavar='M',help='SGD momentum(default:0.5)')
parser.add_argument('--no-cuda',action='store_true',default=False,help='disables CUDA training')
parser.add_argument('--seed',type=int,default=1,metavar='S',help='random seed(default:1)')
parser.add_argument('--log-interval',type=int,default=10,metavar='N',help='how many batches to wait before logging training status')
args=parser.parse_args()
args.cuda=not args.no_cuda and torch.cuda.is_available()

pretransform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5,.5,.5],std=[.5,.5,.5]),
    transforms.Resize([500,500])
])
class PredictSet(data.Dataset):
    def __init__(self,test_root,predict_root):
        fh = open(predict_root, 'r', encoding='utf-8')
        imgs = []
        for line in fh:
            words = line.split(',')
        label = []
        for i in range(len(words) - 1):
            if len(words[i].split('\'')) > 1:
                path = words[i].split('\'')[1]
                imgs.append(path)
        for path in imgs:
            if path.split('data\\\\data\\\\')[1].split(' ')[0] == 'sch':
                label.append(1)
            elif path.split('data\\\\data\\\\')[1].split(' ')[0] == 'ep':
                label.append(0)

        fh = open(test_root, 'r', encoding='utf-8')
        for line in fh:
            words = line.split(',')
        for i in range(len(words)-1):
            if len(imgs) < 64:
                if len(words[i].split('\''))>1:
                    path = words[i].split('\'')[1]
                    imgs.append(path)
        for path in imgs:
            if len(label) < 64:
                if path.split('data\\\\data\\\\')[1].split(' ')[0] == 'sch':
                    label.append(1)
                elif path.split('data\\\\data\\\\')[1].split(' ')[0] == 'ep':
                    label.append(0)

        self.labels=label
        self.imgs=imgs
        self.transforms = pretransform

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

from dataset import TestSet

#对一个模型输出结果
def get_Result(model_path,train_root,test_root,valid_root):

    trainset = SpineSet(train_root)
    validationset = ValidSet(valid_root)

    testset = TestSet(test_root)

    train_loader = DataLoader(dataset=trainset, batch_size=8, shuffle=True)
    valid_loader = DataLoader(dataset=validationset, batch_size=8, shuffle=False)
    test_loader = DataLoader(dataset=testset, batch_size=8, shuffle=False)

    #predictset=PredictSet(test_root,predict_root)

    #predict_loader=DataLoader(dataset=predictset,batch_size=64,shuffle=False)

    device=torch.device("cuda:0" if torch.cuda.is_available() else'cpu')
    #print("第",i,"折: ",device)
    #model= ResNet().cuda()
    model = Inception().cuda()

    if args.cuda:
        model=nn.DataParallel(model).cuda()

    model.load_state_dict(torch.load(model_path))

    model.eval()

    # correct=0
    # total=0
    # with torch.no_grad():
    #     for data in valid_loader:
    #         images,labels=data
    #         images,labels=images.to(device),labels.to(device)
    #         outputs=model(images)
    #
    #         _,predicted=torch.max(outputs.data,dim=1)
    #         total+=labels.size(0)
    #         correct+=(predicted==labels).sum().item()
    #         print(correct,total)

    #print(model)

    device=torch.device("cuda:0" if torch.cuda.is_available() else'cpu')

    #print(predictset.__getitem__(0)[0].reshape((1,3,500,500)))
    # image=predictset.__getitem__(0)[0].reshape((1,3,500,500))
    # label=predictset.__getitem__(0)[1]
    # path=predictset.__getitem__(0)[2]

    #outputs=model(image)
    #print(outputs,label,path)
    count=0
    sum=0
    TP=0
    FP=0
    TN=0
    FN=0
    for data in valid_loader:
        images, labels,paths = data
        images, labels  = images.to(device), labels.to(device)
        outputs = model(images)
        sum=sum+len(labels)
        _, predicted = torch.max(outputs.data, dim=1)
        for i in range(len(labels)):
            #print(paths[i],labels[i],outputs[i],predicted[i] )
            #print(paths[i].split('\\')[5],labels[i].item(), outputs[i], predicted[i].item())
            if labels[i].item()==predicted[i].item():
                count=count+1
                if labels[i].item()==1:
                    TP=TP+1
                else:
                    TN=TN+1
            else:
                if labels[i].item()==1:
                    FN=FN+1
                else:
                    FP=FP+1
    print('Evaluate by img:')
    print(count,sum,count/sum)

    print('TP:',TP,'FP:',FP,'TN:',TN,'FN:',FN)
    print('sensitivity:',TP/(TP+FN),'speci:',TN/(FP+TN))

    path_list=[]
    label_list=[]
    output1prob_list=[]
    TP=0
    FP=0
    TN=0
    FN=0
    for data in valid_loader:
        images, labels,paths = data
        images, labels  = images.to(device), labels.to(device)
        outputs = model(images)
        sum=sum+len(labels)
        _, predicted = torch.max(outputs.data, dim=1)
        for i in range(len(labels)):
            #print(paths[i],labels[i],outputs[i],predicted[i] )
            #print(paths[i].split('\\')[5],labels[i].item(), outputs[i][1].item(), predicted[i].item())
            path_list.append(paths[i].split('\\')[5])
            output1prob_list.append(outputs[i][1].item())
            label_list.append(labels[i].item())

    cutoff=0.5
    #print(np.unique(path_list))
    count_all=0
    sum=0
    for i in range(len(np.unique(path_list))):
        sum=sum+1
        count=0
        output1prob=0
        for j in range(len(path_list)):
            if path_list[j]==np.unique(path_list)[i]:
                count=count+1
                output1prob=output1prob+output1prob_list[j]
                label=label_list[j]
        if output1prob/count>cutoff:
            predict=1
        else:
            predict=0
        if label==predict:
            count_all=count_all+1
            count=count+1
            if label==1:
                TP=TP+1
            else:
                TN=TN+1
        else:
            if label==1:
                FN=FN+1
            else:
                FP=FP+1
    print('Evaluate by patient:')

    print(count_all,sum,count_all/sum)

    print('TP:',TP,'FP:',FP,'TN:',TN,'FN:',FN)
    print('sensitivity:', TP / (TP + FN), 'speci:', TN / (FP + TN), '\n')

#由一个路径找到对应的集合，并输出结果
def get_models_results(model_path):
    type = model_path.split('\\')[-1].split('models_')[0]
    i = int(model_path.split('\\')[-1].split('_fold')[1].split('Epoch')[0])-1
    train_root = "E:\\dwl\\gzw\\2dclassify\\" + type + "\\train_%s.txt" % (i + 1)
    valid_root = "E:\\dwl\\gzw\\2dclassify\\" + type + "\\valid_%s.txt" % (i + 1)
    predict_root = "E:\\dwl\gzw\\2dclassify\\predict.txt"
    test_root = "E:\\dwl\\gzw\\2dclassify\\" + type + "_test.txt"
    print(type, '第', i, '折')
    get_Result(model_path, train_root, test_root, valid_root)

path = 'E:\\dwl\\gzw\\2dclassify\\GZWmodels\\Inception'
modelpath = []
for dirName, subdirList, fileList in os.walk(path):
    for filename in fileList:
        if '.pkl' in filename:
            modelpath.append(os.path.join(dirName, filename))
print(modelpath)
for model_path in modelpath:
    get_models_results(model_path)

model_path="E:\\dwl\\gzw\\2dclassify\\GZWmodels\\Inception\\Cmodels_fold1Epoch100_lr=0.0001bs=64_val_acc=92.98245614035088__test_acc=85.71428571428571_sensitivity=0.925926_specificity=0.933333_.pkl"
