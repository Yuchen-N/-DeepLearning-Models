# 背景：大模型需要大量的计算资源和存储空间，在实际应用却常常不可行。如何能降低模型复杂程度，还保证模型的性能。

## 在相同的数据集，使用不同的模型进行训练，并平均他们的预测，但是很耗费资源，因为算力太过于昂贵而难以部署在用户上，尤其是大型神经网络。
# 大模型形态（训练阶段）Teacher + 小模型形态 （部署阶段）Student-保证低延迟性和小存储空间，计算开销低

## 大模型训练完成后，使用蒸馏 distillation的方式将知识knowledge 转换到容易部署的小模型上面
- __什么是knowledge?__
    - __知识是一个学习好的映射关系，把输入向量映射到输出向量__

## softmax函数 with 温度T

### Softmax Function with Temperature
$$
\text{Softmax}(z_i) = \frac{e^{z_i / T}}{\sum_{j=1}^{n} e^{z_j / T}}
$$


## 准备过程

- ### 准备训练教师网络的数据集
    数据通过教师网络获取logit，通过温度为t的softmax获取soft labels
- ### 构建好学生网络
    让学生网络的logit通过温度为t的softmax和温度为1的soft max，拿到两个结果soft predcitions 和hard predictions。
    __之后让soft labels和soft predictions计算一个distillation loss，让hard predicitons和hard labels计算一个student loss。__
    最终让两个Loss做一个加权和。
- __Hard target 和 soft target__
- __Hard target就是Ground Truth，正样本是1，其他是0; 而soft_target是正样本概率最高，其他类也会有一定的概率，soft_target是由Teacher net prediciton预测结果得到的__


- __why Teacher net prediction target?__
- __教师网络预测结果更准确，同时他会带入到更多的信息在soft target中__

- __教师网络已经有soft labels了，为什么需要hard labels?__
- __虽然教师网络的预测置信度很高，但是不能保证他是完全正确的，ground truth帮助修正soft labels的错误__
