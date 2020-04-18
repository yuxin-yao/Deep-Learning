import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch import autograd, nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
from torch.utils import data
from torchsummaryX import summary
import torch
import torch.nn.functional as F
from torch import autograd, nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
from torch.utils import data
from torchsummaryX import summary
import os
import numpy as np
from PIL import Image

import torch
import torchvision   
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

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



# #model.to(device)
veri = pd.read_csv(r'hw2p2_verification_sample_submission.csv')

veri[['left','right']] = veri.trial.str.split(expand=True) 
  

veri_array= np.array(veri[['left','right']])

model_list = list(model.children())[:-1]
model_list.append(nn.Flatten())
model_1 = nn.Sequential(*model_list)
cos = nn.CosineSimilarity(dim=1, eps=1e-6)


class ImageDataset(Dataset):
    def __init__(self, file_list):
        self.file_list = file_list
        

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img = Image.open(self.file_list[index])
        img = transforms.ToTensor()(img)
        
       
        return img



import os
import time
img_short = os.listdir(r'./test_verification/')
img_list = ['./test_verification/'+img_name for img_name in img_short]


test_dataset = ImageDataset(np.array(img_list))
test_loader_args = dict(shuffle=False, batch_size=256)
test_loader = data.DataLoader(test_dataset, **test_loader_args)


outputresult = np.zeros((len(img_list), 2048))
with torch.no_grad():
    model_1.eval()
    start_time = time.time()

    for batch_idx, data in enumerate(test_loader):
        #data = data.to(device)
        
        outputs = model_1(data)
        
        outputresult[batch_idx*256:(batch_idx+1)*256] = np.array(outputs)
        #print(batch_idx)
        end_time = time.time()
        if (batch_idx%60 == 59):
          print(batch_idx)
          print(outputs.shape)
          print('Time: ',end_time - start_time, 's')

        
        
print('down')
np.save('verinpy', outputresult)




outputresult = np.load('verinpy.npy')
res = {img_short[i]: i for i in range(len(img_short))}
print('down')
#print(res)
from sklearn.metrics.pairwise import cosine_similarity
import time
output = []
start_time = time.time()
for row_num in range(len(veri_array)):
    left_name = veri_array[row_num][0]
    left = outputresult[res[left_name]]
    
    right_name = veri_array[row_num][1]
    right = outputresult[res[right_name]]
 
    a = cosine_similarity(left.reshape(1,-1), right.reshape(1,-1))[0][0]
    end_time = time.time()

    if(row_num % 10000 == 9999):
        print(row_num)
        print('Time: ',end_time - start_time, 's')

    
    output.append(a)




 
import pandas as pd
df = pd.DataFrame(output)

df.to_csv('veri.csv', index=False)
     