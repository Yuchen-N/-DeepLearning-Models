import torch.nn as nn
import torch
from torchvision.transforms import transforms

class Alexnet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

#--------------------------特征提取-------------------------------------#
        self.Conv1 = nn.Conv2d(in_channels=3, out_channels= 96, kernel_size= 11, stride = 4, padding= 2)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1= nn.MaxPool2d(kernel_size= 3, stride= 2)
        self.Conv2 = nn.Conv2d(in_channels = 96, out_channels =256, kernel_size=5, padding= 2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.Conv3 = nn.Conv2d(in_channels= 256, out_channels=384, kernel_size= 3, padding= 1)
        self.Conv4 = nn.Conv2d(in_channels= 384, out_channels= 384, kernel_size= 3, padding= 1)
        self.Conv5 = nn.Conv2d(in_channels = 384, out_channels= 256, kernel_size= 3, padding= 1)
        self.maxpool3 =nn.MaxPool2d(kernel_size=3, stride= 2)
#------------------------------分类器-------------------------------------#
        self.dropout = nn.Dropout()
        self.fc1 = nn.Linear(256*1*1, 512)
        self.fc2 = nn.Linear(512,512)
        self.output = nn.Linear(512, num_classes)
    def forward(self, x):
        x = self.Conv1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.Conv2(x)
        x = self.relu(x)
        x = self.maxpool2(x)
        x = self.Conv3(x)
        x = self.relu(x)
        x = self.Conv4(x)
        x = self.relu(x)
        x = self.Conv5(x)
        x = self.relu(x)
        x = self.maxpool3(x)
#-----------------------------分类器---------------------------------------#
        # 现在图像特征已经提取完成，获得了特征的情况下，
        #我们对x进行展平操作，对已经有特征的图像找里面有特征的数据
        x = torch.flatten(x, 1) 
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.output(x)
        return x
