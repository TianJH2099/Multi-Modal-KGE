## MultiModel Knowledge Graph Embeding

* ResNet50+KGE Model

-----
ResNet50图片特征提取最后一层需要修改，学习方式需要重新设计
目前ConvE、ConvR和TuckER实验，但是数据量过大，容易陷入局部最优

Adam优化器容易陷入局部最优 Loss:3.
SGD能够将Loss降至1以下，目前最多训练了200+epoch，最低loss:0.74

------
_将数据集放置如下格式_
------
* OpenBG-IMG
> OpenBG-IMG_images
>> ent_000000
>>> iamge_0.jpg