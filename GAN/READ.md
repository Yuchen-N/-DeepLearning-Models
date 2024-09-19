# PyTorch DCGAN 训练脚本
这是一个使用 PyTorch 实现的生成对抗网络（GAN）训练脚本，用于从多个数据集（如 CIFAR-10 和 MNIST）生成图像。该脚本通过命令行参数提供了高度的灵活性，允许用户自定义模型参数、训练配置和数据集选择。

## 环境要求

- **Python**：3.7 或更高版本
- **PyTorch**：1.7.0 或更高版本
- **设备支持**：
  - CUDA（GPU）
  - MPS（Apple Silicon）
  - CPU 回退

# 模型架构

- ## 生成器使用一系列转置卷积层（ConvTranspose2d）将潜在向量（z）上采样成图像。每层后面跟随批归一化（BatchNorm2d）和 ReLU 激活函数，以稳定训练并引入非线性

```python
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

```
- ## 判别器使用卷积层（Conv2d）将输入图像下采样，并最终输出一个概率值，表示输入图像是真实的还是生成的。每层后面跟随批归一化（BatchNorm2d）和 LeakyReLU 激活函数，以改善梯度流动和稳定性。
```python
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
```

- ## 输出结果
- ### 生成的图像：保存在指定的 output_dir 目录下，文件名格式为 fake_samples_epoch_{epoch}.png。
