# AlexNet

该目录包含 AlexNet 的 PyTorch 实现，原始论文：[ImageNet Classification with Deep Convolutional Neural Networks]

## 文件结构

- `AlexModel.py`：定义了 AlexNet 模型的架构代码。
- `train.py`：训练模型的脚本，包括训练过程和参数设置。
- `eval.py`：评估模型性能的脚本，用于在测试集上计算准确率等指标。
- `imageFolder_Dataset.py`：用于加载训练数据集的脚本，基于目录结构的数据集。
- `txt_Dataset.py`：用于加载测试数据集的脚本，基于文本文件列表的数据集
