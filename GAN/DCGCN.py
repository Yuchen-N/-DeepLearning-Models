from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as utils

# 设置命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | lsun | mnist | imagenet | folder | lfw | fake')
parser.add_argument('--dataroot', required=False, help='path to dataset')
parser.add_argument('--workers', type=int, default=2, help='number of data loading workers')
parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
parser.add_argument('--img_size', type=int, default=64, help='input image size for the network')
parser.add_argument('--latent_dim', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--n_filters_gen', type=int, default=64)
parser.add_argument('--n_filters_disc', type=int, default=64)
parser.add_argument('--epochs', type=int, default=25, help='number of epochs to train')
parser.add_argument('--learning_rate', type=float, default=0.0002, help='learning rate')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
parser.add_argument('--use_cuda', action='store_true', help='whether to use CUDA')
parser.add_argument('--output_dir', default='output', help='directory for output images and model checkpoints')
parser.add_argument('--seed', type=int, help='manual seed for reproducibility')
parser.add_argument('--mps', action='store_true', default=False, help='enable macOS MPS training')

opt = parser.parse_args()
print(opt)

# 设置随机种子
if opt.seed is None:
    opt.seed = random.randint(1, 10000)
print(f"Using random seed: {opt.seed}")
random.seed(opt.seed)
torch.manual_seed(opt.seed)

# 确认输出文件夹
os.makedirs(opt.output_dir, exist_ok=True)

# 检查设备
device = torch.device('cuda:0' if torch.cuda.is_available() and opt.use_cuda else 'mps' if opt.mps else 'cpu')
print(f"Using device: {device}")

# 数据加载
if opt.dataset == 'cifar10':
    dataset = datasets.CIFAR10(root=opt.dataroot, download=True, transform=transforms.Compose([
        transforms.Resize(opt.img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ]))
    nc = 3
elif opt.dataset == 'mnist':
    dataset = datasets.MNIST(root=opt.dataroot, download=True, transform=transforms.Compose([
        transforms.Resize(opt.img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ]))
    nc = 1
else:
    raise ValueError("Dataset not supported")

dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers)

# 定义生成器
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(opt.latent_dim, opt.n_filters_gen * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(opt.n_filters_gen * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(opt.n_filters_gen * 8, opt.n_filters_gen * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.n_filters_gen * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(opt.n_filters_gen * 4, opt.n_filters_gen * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.n_filters_gen * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(opt.n_filters_gen * 2, opt.n_filters_gen, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.n_filters_gen),
            nn.ReLU(True),
            nn.ConvTranspose2d(opt.n_filters_gen, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)

# 定义判别器
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, opt.n_filters_disc, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(opt.n_filters_disc, opt.n_filters_disc * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.n_filters_disc * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(opt.n_filters_disc * 2, opt.n_filters_disc * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.n_filters_disc * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(opt.n_filters_disc * 4, opt.n_filters_disc * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.n_filters_disc * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(opt.n_filters_disc * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x).view(-1)

# 模型初始化
netG = Generator().to(device)
netD = Discriminator().to(device)

# 初始化权重
def init_weights(m):
    if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.constant_(m.bias, 0)

netG.apply(init_weights)
netD.apply(init_weights)

# 损失函数与优化器
criterion = nn.BCELoss()
optimizerD = optim.Adam(netD.parameters(), lr=opt.learning_rate, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.learning_rate, betas=(opt.beta1, 0.999))

# 噪声向量
fixed_noise = torch.randn(opt.batch_size, opt.latent_dim, 1, 1, device=device)

# 训练循环
for epoch in range(opt.epochs):
    for i, data in enumerate(dataloader):
        # 训练判别器
        netD.zero_grad()
        real_data = data[0].to(device)
        batch_size = real_data.size(0)
        label_real = torch.full((batch_size,), 1, dtype=torch.float, device=device)
        label_fake = torch.full((batch_size,), 0, dtype=torch.float, device=device)

        output_real = netD(real_data)
        errD_real = criterion(output_real, label_real)

        noise = torch.randn(batch_size, opt.latent_dim, 1, 1, device=device)
        fake_data = netG(noise)
        output_fake = netD(fake_data.detach())
        errD_fake = criterion(output_fake, label_fake)

        errD = errD_real + errD_fake
        errD.backward()
        optimizerD.step()

        # 训练生成器
        netG.zero_grad()
        output_fake = netD(fake_data)
        errG = criterion(output_fake, label_real)  # 生成器的目标是让D判别其生成的图像为真
        errG.backward()
        optimizerG.step()

        # 日志输出
        print(f'Epoch [{epoch}/{opt.epochs}] Step [{i}/{len(dataloader)}] Loss_D: {errD.item()} Loss_G: {errG.item()}')

    # 保存图片和模型
    fake_images = netG(fixed_noise).detach()
    utils.save_image(fake_images, f'{opt.output_dir}/fake_samples_epoch_{epoch}.png', normalize=True)

    torch.save(netG.state_dict(), f'{opt.output_dir}/netG_epoch_{epoch}.pth')
    torch.save(netD.state_dict(), f'{opt.output_dir}/netD_epoch_{epoch}.pth')
