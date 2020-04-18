

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


import torch
import torch.nn.functional as F
from torch import autograd, nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
from torch.utils import data
from torchsummaryX import summary

#test_dataset = datasets.ImageFolder(root=r'./test_classification', transform=transforms.Compose([transforms.Resize(128),transforms.ToTensor()]))
test_dataset = datasets.ImageFolder(root=r'./test_classification', transform=transforms.ToTensor())


num_workers = 4



test_loader_args = dict(shuffle=False, batch_size=1)
test_loader = data.DataLoader(test_dataset, **test_loader_args)

class BasicBlock(nn.Module):

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
       
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample
        
    def forward(self, x):
        
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet34(nn.Module):

    def __init__(self, num_classes):
        super(ResNet34, self).__init__()
        
        block = BasicBlock
        
        self.repeat_layers = []
        self.repeat_layers.append(nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False))
        self.repeat_layers.append(nn.BatchNorm2d(64))
        self.repeat_layers.append(nn.ReLU(inplace=True))

        out_channel_list = [64, 128, 256, 512]
        blocks_list = [3, 4, 6, 3]
        stride_list = [1, 2, 2, 2]
        for i in range(4):
            downsample = None
            if i == 0:
                in_channel_temp = 64
            else:
                in_channel_temp = out_channel_list[i-1]
        
            if stride_list[i] != 1:
                downsample = nn.Sequential(
                    nn.Conv2d(in_channel_temp, out_channel_list[i], kernel_size=1, stride=stride_list[i], bias=False),
                    nn.BatchNorm2d(out_channel_list[i]),
                )


            self.repeat_layers.append(block(in_channel_temp, out_channel_list[i], stride_list[i], downsample))
            
            for _ in range(1, blocks_list[i]):
                self.repeat_layers.append(block(out_channel_list[i], out_channel_list[i]))
                
        self.repeat_layers.append(nn.AdaptiveAvgPool2d((2, 2)))
        self.repeat_layers = nn.Sequential(*self.repeat_layers)    
        
        self.linear_out = nn.Linear(512 * 4, num_classes)

 

    def forward(self, x):
        
        output = x
        output = self.repeat_layers(output)
        output = torch.flatten(output, 1)
        output = self.linear_out(output)

        return output


model = ResNet34(2300)

model.load_state_dict(torch.load('hw2_2_resnet.pth'))
model.eval()


device = torch.device("cuda" if cuda else "cpu")
#model.to(device)

outputresult = []
with torch.no_grad():
    model.eval()
    for batch_idx, (data, label) in enumerate(test_loader):
        #data = data.to(device)
        #print(data)
        
        outputs = model(data)
        if (batch_idx%100) == 99:
            print(batch_idx)

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
