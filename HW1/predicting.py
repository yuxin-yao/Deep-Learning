# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 16:26:01 2020

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



#test_xx = np.load("test.npy",allow_pickle=True)

print('data')

#test_xx = np.concatenate(test_xx).reshape(-1,40)

print('data')
class MyDataset(data.Dataset):
    def __init__(self, X):
        self.X = X
       

    def __len__(self):
        return len(self.X)

    def __getitem__(self,index):
        X = np.take(self.X,np.array(range(index-12,index+13,1)),axis=0, mode = 'clip').float().reshape(-1) #flatten the input

       
        return X

num_workers = 4


#np.save('test_test_xx',test_xx)
test_xx = np.load('test_test_xx_pca.npy',allow_pickle=True)
#
print(test_xx.shape)


test_xx =  torch.from_numpy(test_xx)

test_dataset = MyDataset(test_xx)


test_loader_args = dict(shuffle=False, batch_size=1)
test_loader = data.DataLoader(test_dataset, **test_loader_args)

print('data')
print('data')

print('data')


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def make_divisible(x, divisible_by=8):
    import numpy as np
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)




input_size=23
width_mult=1.0
block = InvertedResidual
input_channel = 32
last_channel = 1280
n_class=2300
interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer

        # input_channel = make_divisible(input_channel * width_mult)  # first channel is always 32!
last_channel = make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel
features = [conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
for t, c, n, s in interverted_residual_setting:
    output_channel = make_divisible(c * width_mult) if t > 1 else c
    for i in range(n):
        if i == 0:
            features.append(block(input_channel, output_channel, s, expand_ratio=t))
        else:
            features.append(block(input_channel, output_channel, 1, expand_ratio=t))
        input_channel = output_channel
        # building last several layers
features.append(conv_1x1_bn(input_channel, last_channel))
        # make it nn.Sequential
features.append(nn.AvgPool2d(2))
features.append(nn.Dropout(p=0.2, inplace=False))
features.append(nn.Flatten())
features.append(nn.Linear(5120, n_class))
model = nn.Sequential(*features)

model.load_state_dict(torch.load('hw2_2_mobile_upscale_16.pth'))
model.eval()


device = torch.device("cuda" if cuda else "cpu")
#model.to(device)
print(model)
outputresult = []
with torch.no_grad():
    model.eval()
    for batch_idx, (data) in enumerate(test_loader):
        #data = data.to(device)
        outputs = model(data)

        _, predicted = torch.max(outputs.data, 1)
        outputresult.append(predicted)
      
        outputresult.append(outputs)

import pickle
with open('predict_mobilenet_upscale.pkl', 'wb') as f:
    pickle.dump(outputresult, f)


a= [outputresult[i] for i in range(len(outputresult)) if i%2==0]
b= [a[i].item() for i in range(len(a))]

import pandas as pd
df = pd.DataFrame(b)

df.to_csv('predict_mobilenet_upscale.csv', index=False)
