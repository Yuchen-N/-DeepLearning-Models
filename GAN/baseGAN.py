import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision.transforms import transforms

#要先对数据进行归一化
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.5,0.5)
])

#实例化torchvision.datasets.MNIST类，下载MNIST手写数字数据集
train_ds = torchvision.datasets.MNIST('data',
                                      train=True,
                                      transform=transform,
                                      download=True)

dataloader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)

#生成器：特征长度为100的噪声（正态分布随机数）
#tensor大小为：batch_size * 100
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(100, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 28*28)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
    
    def forward(self, x):       #x是noise输入
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.tanh(x)
        generate_image = x.view(-1, 1, 28,28)
        return generate_image
    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512,256)
        self.fc3 = nn.Linear(256,1)
        self.sigmoid = nn.Sigmoid()
        self.Leakyrelu = nn.LeakyReLU()
    def forward(self, x):
        x = x.view(-1,28*28)
        x = self.fc1(x)
        x = self.Leakyrelu(x)
        x = self.fc2(x)
        x = self.Leakyrelu(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x
    
gen = Generator()
dis = Discriminator()

d_optim = torch.optim.Adam(dis.parameters(), lr=0.0001)
g_optim = torch.optim.Adam(gen.parameters(),lr = 0.0001)

loss_fun =torch.nn.BCELoss

#---------绘图函数-------------------
def gen_img_plot(model, epoch, test_input):

    #将CUDA tensor格式的数据转换成numpy
    #需要先转换成cpu float-tensor随后再转换成numpy格式，numpy不能读取CUDA tensor需要先转换为CPU tensor
    prediction = np.squeeze(model(test_input).detach().cpu().numpy())
    fig = plt.figure(figsize=(4,4))
    for i in range(16):
        plt.subplot(4,4,i+1)
        plt.imshow((prediction[i]+1)/2)
        plt.axis('off')
    plt.show()

test_input = torch.randn(16,100)

D_loss = []
G_loss = []
for epoch in range(20):
    d_epoch_loss = 0
    g_epoch_loss = 0
    count = len(dataloader)
    for step, (img,_) in enumerate(dataloader):
        img = img.to(device)