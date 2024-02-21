import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data import SpineSet
from model import Inception

def train(epoch):
    running_loss=0.0
    for batch_idx,data in enumerate(train_loader,0):
        inputs,target=data
        #inputs,target=inputs.to(device),target.to(device)
        optimizer.zero_grad()

        outputs=model(inputs)
        loss=criterion(outputs,target)
        loss.backward()
        optimizer.step()

        print('Epoch:',epoch,'Loss:',loss.item())
        running_loss+=loss.item()
        if batch_idx%300==299:
            print('[%d,%5d]loss:%3f'%(epoch+1,batch_idx+1,running_loss/300))
            running_loss=0
    return loss.item()

def validation():
    correct=0
    total=0
    with torch.no_grad():
        for data in valid_loader:
            images,labels=data
            #images, labels=images.to(device),labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            _,predicted=torch.max(outputs.data,dim=1)
            total+=labels.size(0)
            correct+=(predicted==labels).sum().item()
    print('Accuracy on valid set:%d %%'%(100*correct/total))
    print('Loss on valid set:',loss.item())
    return loss.item(),100*correct/total

train_root=".\\trainpath.txt"
valid_root=".\\validpath.txt"

trainset=SpineSet(train_root)
validset=SpineSet(valid_root)

train_loader=DataLoader(dataset=trainset,batch_size=64,shuffle=True)
valid_loader=DataLoader(dataset=validset,batch_size=64,shuffle=False)

device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#model=Inception().cuda()
model=Inception()
model=nn.DataParallel(model).cuda()

optimizer=optim.SGD(model.parameters(),lr=0.001,momentum=0.5)
criterion=nn.CrossEntropyLoss()
for epoch in range(100):
    train_loss=train(epoch)
    val_loss,val_acc=validation()
    torch.save(model.state_dict(),'.\\Epoch%s'%(i+1)+'train_acc==%s'%(val_acc)+'.pkl')
