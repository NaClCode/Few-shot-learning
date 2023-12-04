# FEAT
## Title 
Few-Shot Learning via Embedding Adaptation with Set-to-Set Functions
## Abstract
在本文中，我们提出了一种新的方法，`通过集合到集合的函数使实例嵌入适应目标分类任务，从而产生特定于任务的、具有区别性的嵌入`。我们根据经验研究了这种set-to-set函数的各种实例，并观察到`Transformer是最有效的`——因为它自然地满足了我们期望模型的关键属性。我们将这个模型表示为FEAT (few-shot embedding adaptation Transformer)，并在标准的few-shot分类基准和四个扩展的few-shot学习设置上验证它，这些设置具有基本的用例，即跨域、换向、广义的few-shot学习和low-shot学习。
## Introduction

### Few-Shot learning
### 不同
- 传统机器学习: 让模型"认识"图片, 再泛化到数据集
- Few-Shot learning: 让模型区分两个图片相似性
### 两种实现
- 直接使用图片通过CNN或者ResNet提取特征，然后使用简单分类器进行分类
- 首先用CNN或者ResNet提取特征，然后使用Set-to-Set函数提取特征，最后使用简单分类器进行分类
#### 常见术语
N-way: 分类有N个类别
M-shot: 每个类别下只训练M张图片
support sets/training set: 训练集

### Set-to-Set functions
这种基于模型的嵌入自适应需要一个集合到集合的函数:一个函数映射，它从少量支持集中获取所有实例，并输出自适应的支持实例嵌入集，集合中的元素相互自适应。然后将这些输出嵌入组装为每个视觉类别的原型，并作为最近邻分类器。
本文论证了Transformer是这种集合到集合函数的最佳选择。

## Material and method

### 实验数据集
#### MiniImageNet
- 100个类, 600个示例
- 64类作为SEEN类别训练
- 16与20类作为UNSEEN类别进行模型验证与评估
#### TieredImageNet
- 160个类, 35197个示例
- 训练, 验证, 评估
#### OFficeHome
- 8722图片, 25个类训练, 15, 25个类进行评估
- 验证跨域泛化能力

### 实验
- 比较直接嵌入: 使用了set to set 调整效果更好
- 比较骨架网络: ResNet 效果更好
- 比较set to set 函数: 直接使用Transformer效果更好
## Results and discussion
当有少量标记的训练数据时。我们建议使用set-to-set函数进行嵌入自适应，并使用transformer (FEAT)作为set to set 函数
## 启发
- 把论文的transformer 换成vit
- 修改骨架网络 