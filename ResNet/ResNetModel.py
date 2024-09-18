import torch.nn as nn
import torch

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=inplanes, out_channels= planes, kernel_size= 3, stride= stride, padding= 1,groups = groups, dilation= dilation, bias = False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=planes, out_channels= planes, kernel_size= 3, stride= 1, padding= 1, groups = groups, dilation= dilation, bias = False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        identity = x # 这里就是这个Block接受的输入

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        #将上一层的输入和这一层的输出特征图相加作为传入给下一层的特征图

        out = self.relu(out)
        
        return out
    

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes = 200, groups = 1, width_per_group = 64,replace_stride_with_dilation=None,
                 norm_layer=None):
        super().__init__()
        self.inplanes = 64
        self.groups = groups
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size = 3, stride = 1, padding =1, bias = False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace = True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, padding= 1, stride= 1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride = 1)
        self.layer3 = self._make_layer(block, 256, layers[2], stride = 2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride = 2)

        self.avepool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*BasicBlock.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = nn.BatchNorm2d
        downsample = None

        # 如果需要调整特征图的尺寸或通道数，定义 downsample
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avepool(x)
        x = torch.flatten(x,1)
        x = self.fc(x)
        return x
