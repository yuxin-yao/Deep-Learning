# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 14:36:44 2020

@author: genev
"""



import numpy as np
import torch
import sys
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from torchvision import transforms
from torchvision.datasets import MNIST

import matplotlib.pyplot as plt
import time

cuda = torch.cuda.is_available()
cuda
import sklearn.model_selection



train_xx = np.load("train.npy",allow_pickle=True)
train_yy = np.load("train_labels.npy",allow_pickle=True)

test_xx = np.load("dev.npy",allow_pickle=True)
test_yy = np.load("dev_labels.npy",allow_pickle=True)

test_test_xx = np.load("test.npy",allow_pickle=True)


train_yy = np.concatenate(train_yy)
train_xx = np.concatenate(train_xx).reshape(-1,40)


test_yy = np.concatenate(test_yy)
test_xx = np.concatenate(test_xx).reshape(-1,40)

test_test_xx = np.concatenate(test_test_xx).reshape(-1,40)


class MyDataset(data.Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.Y)

    def __getitem__(self,index):
        X = np.take(self.X,np.array(range(index-12,index+13,1)),axis=0, mode = 'clip').float().reshape(-1) #flatten the input

        Y = self.Y[index].long()
        return X,Y

num_workers = 4

#np.save('test_yy',test_yy)
#np.save('test_xx',test_xx)

# Training
#train_yy = np.load('train_yy.npy',allow_pickle=True)
#train_xx = np.load('train_xx.npy',allow_pickle=True)
#test_yy = np.load('test_yy.npy',allow_pickle=True)
#test_xx = np.load('test_xx.npy',allow_pickle=True)

#test_test_xx = np.load('test_test_xx.npy',allow_pickle=True)

from sklearn.decomposition import PCA

pca = PCA(n_components=20)
#print(train_xx.shape)

pca.fit(train_xx[0:500000])
print(pca.explained_variance_ratio_)
train_xx = pca.transform(train_xx)
test_xx = pca.transform(test_xx)
test_test_xx = pca.transform(test_test_xx)


np.save('train_xx_pca',train_xx)
np.save('test_xx_pca',test_xx)
np.save('test_test_xx_pca',test_test_xx)

train_yy =  torch.from_numpy(train_yy)
train_xx =  torch.from_numpy(train_xx)

test_yy =  torch.from_numpy(test_yy)
test_xx =  torch.from_numpy(test_xx)

print('data')
train_dataset = MyDataset(train_xx, train_yy)
# Shuffle the data when training
train_loader_args = dict(shuffle=True, batch_size=256, num_workers=num_workers, pin_memory=True) if cuda\
                    else dict(shuffle=True, batch_size=64)
train_loader = data.DataLoader(train_dataset, **train_loader_args)                                                      
print('data')
# Testing
test_dataset = MyDataset(test_xx, test_yy)

test_loader_args = dict(shuffle=False, batch_size=256, num_workers=num_workers, pin_memory=True) if cuda\
                    else dict(shuffle=False, batch_size=1)
test_loader = data.DataLoader(test_dataset, **test_loader_args)
print('data')


class Simple_MLP(nn.Module):
    def __init__(self, size_list):
        super(Simple_MLP, self).__init__()
        layers = []
        self.size_list = size_list
        for i in range(len(size_list) - 2):
            if i%2 == 0:
                layers.append(nn.Linear(size_list[i],size_list[i+1]))
                layers.append(nn.BatchNorm1d(size_list[i+1]))
                layers.append(nn.Dropout(p=0.2))
                layers.append(nn.PReLU())
            else:
                layers.append(nn.Linear(size_list[i],size_list[i+1]))
                layers.append(nn.BatchNorm1d(size_list[i+1]))
               # layers.append(nn.Dropout(p=0.1))
                layers.append(nn.PReLU())
        layers.append(nn.Linear(size_list[-2], size_list[-1]))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

model = Simple_MLP([500, 2048, 1800, 1448, 1024, 724, 424, 138])
#28*28 = 784
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr = 1e-3)
device = torch.device("cuda" if cuda else "cpu")
model.to(device)
print(model)

def train_epoch(model, train_loader, criterion, optimizer):
    #important: model.train
    model.train()

    running_loss = 0.0

    start_time = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()   # .backward() accumulates gradients          !!!!! clear the gradients after backward
        data = data.to(device)
        target = target.to(device) # all data & model on same device

        outputs = model(data)
        loss = criterion(outputs, target)
        running_loss += loss.item()

        loss.backward()
        optimizer.step()

    end_time = time.time()

    running_loss /= len(train_loader)
    print('Training Loss: ', running_loss, 'Time: ',end_time - start_time, 's')
    return running_loss



def test_model(model, test_loader, criterion):
    with torch.no_grad():
        model.eval()

        running_loss = 0.0
        total_predictions = 0.0
        correct_predictions = 0.0

        for batch_idx, (data, target) in enumerate(test_loader):
            data = data.to(device)
            target = target.to(device)

            outputs = model(data)

            _, predicted = torch.max(outputs.data, 1)
            total_predictions += target.size(0)
            correct_predictions += (predicted == target).sum().item()

            loss = criterion(outputs, target).detach()
            running_loss += loss.item()



        running_loss /= len(test_loader)
        acc = (correct_predictions/total_predictions)*100.0
        print('Testing Loss: ', running_loss)
        print('Testing Accuracy: ', acc, '%')
        return running_loss, acc

n_epochs = 5
Train_loss = []
Test_loss = []
Test_acc = []

print('train start')
for i in range(n_epochs):
    train_loss = train_epoch(model, train_loader, criterion, optimizer)
    test_loss, test_acc = test_model(model, test_loader, criterion)
    Train_loss.append(train_loss)
    Test_loss.append(test_loss)
    Test_acc.append(test_acc)
    print('='*20)
    
    
# Print model's state_dict
#print("Model's state_dict:")
#for param_tensor in model.state_dict():
#    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

# Print optimizer's state_dict
#print("Optimizer's state_dict:")
#for var_name in optimizer.state_dict():
#    print(var_name, "\t", optimizer.state_dict()[var_name])
torch.save(model.state_dict(),'checkpointuntitled3_Prelu.pth')
