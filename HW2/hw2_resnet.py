import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.functional as F
from torch import autograd, nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
from torch.utils import data
from torchsummaryX import summary

train_dataset = datasets.ImageFolder(root=r'./train_data/medium', transform=transforms.ToTensor())

test_dataset = datasets.ImageFolder(root=r'./validation_classification/medium', transform=transforms.ToTensor())



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


device = torch.device("cuda")
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
criterion = nn.CrossEntropyLoss()

#optimizer = optim.Adam(model.parameters(),lr = 0.01)
# optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01, alpha=0.9, weight_decay=0.00004, momentum=0.9)
# scheduler = StepLR(optimizer, 1, gamma=0.98)
optimizer = torch.optim.SGD(model.parameters(), lr=0.045, momentum=0.9, weight_decay=0.0001, nesterov=True)
scheduler = StepLR(optimizer, 1, gamma=0.98)
# from torch.optim.lr_scheduler import ReduceLROnPlateau
# scheduler = ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=2, verbose=True,
#                                   threshold_mode='abs', threshold=0.01, min_lr=1e-6)


train_loader_args = dict(shuffle=True, batch_size=256)
train_loader = data.DataLoader(train_dataset, **train_loader_args)

test_loader_args = dict(shuffle=True, batch_size=64)
test_loader = data.DataLoader(test_dataset, **train_loader_args)

closs_weight = 1
model.to(device)
import time
def train_epoch(model, train_loader, criterion, optimizer):
    #important: model.train
    model.train()

    running_loss = 0.0
    total_predictions = 0.0
    correct_predictions = 0.0
    
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
        _, predicted = torch.max(outputs.data, 1)
        total_predictions += target.size(0)
        correct_predictions += (predicted == target).sum().item()
    
    end_time = time.time()
    
    running_loss /= len(train_loader)
    acc = (correct_predictions/total_predictions)*100.0
        
    print('Training Loss: ', running_loss, 'Time: ',end_time - start_time, 's')
    print('Training Accuracy: ', acc, '%')
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
n_epochs = 100
Train_loss = []
Test_loss = []
Test_acc = []






for i in range(n_epochs):
    print(i+1, 'epoch')
    for param_group in optimizer.param_groups:
        print(param_group['lr'])

    train_loss = train_epoch(model, train_loader, criterion, optimizer)
    test_loss, test_acc = test_model(model, test_loader, criterion)
    Train_loss.append(train_loss)
    Test_loss.append(test_loss)
    Test_acc.append(test_acc)
    print('='*20)
    a = 'hw2_2_resnet_'+'_'+str(i)+'.pth'
    print(a)
    torch.save(model.state_dict(),a)
    scheduler.step(test_acc)
    if i < 15:
        optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr']*0.9



torch.save(model.state_dict(),'hw2_2_resnet.pth')
