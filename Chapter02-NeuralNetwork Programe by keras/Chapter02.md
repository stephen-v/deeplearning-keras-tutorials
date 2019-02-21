- [1. keras 神经网络编程基础](#1-keras-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E7%BC%96%E7%A8%8B%E5%9F%BA%E7%A1%80)
  - [1.1. 神经网络基本概念](#11-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%9F%BA%E6%9C%AC%E6%A6%82%E5%BF%B5)
    - [1.1.1. 层](#111-%E5%B1%82)
    - [1.1.2. 损失函数](#112-%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0)
      - [1.1.2.1. 均方差损失函数的不足](#1121-%E5%9D%87%E6%96%B9%E5%B7%AE%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0%E7%9A%84%E4%B8%8D%E8%B6%B3)
      - [1.1.2.2. 交叉熵损失函数](#1122-%E4%BA%A4%E5%8F%89%E7%86%B5%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0)
    - [1.1.3. 训练模型](#113-%E8%AE%AD%E7%BB%83%E6%A8%A1%E5%9E%8B)
# 1. keras 神经网络编程基础
## 1.1. 神经网络基本概念
回顾前面所讲的内容神经网络有几个部分构成，其关系可由下图表示：
* **层**，多个层组合成网络（或模型）
* **输入数据**，原始数据
* **损失函数**，预测值与实际值评价函数
* **优化器**，更新权重的方法
![2019-02-20-15-39-46](http://www.xdpie.com/2019-02-20-15-39-46.png)


现在通过一个实际的例子来讲解神经网络的基础组成部分，这里要解决的问题是，将手写数字的灰度图像（28 像素×28 像素）划分到 10 个类别中（0~9）。我们将使用 MNIST 数据集如下图所示，它是机器学习领域的一个经典数据集，其历史几乎和这个领域一样长，而且已被人们深入研究。这个数据集包含 60 000 张训练图像和 10 000 张测试图像，由美国国家标准与技术研究院（National Institute of Standards and Technology，即 MNIST 中的 NIST）在 20 世纪 80 年代收集得到。

![2019-02-20-16-34-17](http://www.xdpie.com/2019-02-20-16-34-17.png)

在原始数据中给出的是(60000,28,28)的3d张量作为训练集，(10000,28,28)的张量作为测试集。我们设计的神经网络如下：
![2019-02-20-16-25-27](http://www.xdpie.com/2019-02-20-16-25-27.png)

因此针对minist数据进行预处理，将(60000,28,28)转换为(60000,28*28)的张量，并缩放到所有值都在 [0, 1] 区间。比如，之前训练图像保存在一个 uint8 类型的数组中，其形状为(60000, 28, 28) ，取值区间为 [0, 255] 。我们需要将其变换为一个 float32 数组，其形状为 (60000, 28 * 28) ，取值范围为 0~1




### 1.1.1. 层
层是一个数据处理模块，将一个或多个输入张量转换为一个或多个输出张量。有些层是无状态的，但大多数的层是有状态的，即层的权重。权重是利用随机梯度下降学到的一个或多个张量，其中包含网络的知识。不同的张量格式与不同的数据处理类型需要用到不同的层。例如，简单的向量数据保存在形状为 (samples, features) 的 2D 张量中，通常用密集连接层［densely connected layer，也叫全连接层（fully connected layer）或密集层（dense layer），对应于 Keras 的 Dense 类］来处理。序列数据保存在形状为 (samples, timesteps, features) 的 3D 张量中，通常用循环层（recurrent layer，比如 Keras 的 LSTM 层）来处理。图像数据保存在 4D 张量中，通常用二维卷积层（Keras 的 Conv2D ）来处理。你可以将层看作深度学习的乐高积木，Keras 等框架则将这种比喻具体化。在 Keras 中，构建深度学习模型就是将相互兼容的多个层拼接在一起，以建立有用的数据变换流程。这里层兼容性（layer compatibility）具体指的是每一层只接受特定形状的输入张量，并返回特定形状的输出张量。
```Python
from keras import models
from keras import layers
from keras.utils import plot_model

network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))

plot_model(network, show_layer_names=True, show_shapes=True, to_file='./plot_model.png')

```

### 1.1.2. 损失函数
在前一章说明了损失函数指导着网络朝着正确的方向行进，损失函数有许多种之前我们采用均方差作为损失函数，这里我们将采用交叉熵函数来代替。
#### 1.1.2.1. 均方差损失函数的不足
一般来说均方差损失函数会出现学习缓慢的问题，在前面的推导中我们已经知道偏导数决定学习速率因此学习速度慢也就是偏导数小。
$$
C = \frac{(y-\hat{y})^2}{2} \tag*{2.1}
$$
其中$\hat{y}$是神经元的输出。显式地使⽤权重和偏置来表达这个，我们有 $ \hat{y}= sigmod(z)$，其中 z = wx + b。使⽤链式法则来求权重和偏置的偏导数就有

$$
\frac{\partial{C}}{\partial{w}} = (\hat{y} - y )sigmoid(z)^\prime x \tag*{2.2}
$$

$$
\frac{\partial{C}}{\partial{b}} = (\hat{y} - y )sigmoid(z)^\prime \tag*{2.3}
$$

可以看到偏导数都与sigmoid函数的倒数有关，而sigmoid函数在越接近1时越平缓倒数越小，这也就解释了为什么学习缓慢的原因。
#### 1.1.2.2. 交叉熵损失函数
我们如下定义这个神经元的交叉熵损失函数：
$$
C=-\frac{1}{n} \sum\limits_{x}[yln\hat{y} + (1-y)ln(1-\hat{y})] \tag*{2.4} 
$$
如果对于所有的训练输⼊ x，神经元实际的输出接近⽬标值，那么交叉熵将接近 0,假设在这个例⼦中，y = 0 ⽽ $\hat{y}$ ≈ 0。这是我们想到得到的结果,由于y=0则中括号中第一项消去，由于$\hat{y}$ ≈ 0则第二项无限接近与0.

对式(2.4)求偏导数化简得到如下：  

$$
\frac{\partial{C}}{\partial{w}} = \frac{1}{n} \sum\limits_{x} x_j(sigmoid(z) - y) \tag*{2.5}
$$

这是⼀个优美的公式。它告诉我们权重学习的速度受到 $sigmoid(z) − y$，也就是输出中的误差的控
制。更⼤的误差，更快的学习速度。这是我们直觉上期待的结果。特别地，这个代价函数还避免
了像在⼆次代价函数中类似⽅程中 $sigmoid\prime (z)$ 导致的学习缓慢.

### 1.1.3. 训练模型
```Python
  network = get_model()
    network.compile(optimizer='sgd',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    train_images, test_images, train_labels, test_labels = preprocess()
    network.fit(train_images, train_labels, epochs=5, batch_size=128)
    test_loss, test_acc = network.evaluate(test_images, test_labels)
    print('test_acc:', test_acc)
```
在模型编译时我们采用sgd(随机梯度优化)和交叉熵损失函数，在5轮后达到test_acc: 0.9133的准确率，这个数字并不是很好，keras提供了其他更为优秀的优化器可以在同样训练次数的情况下达到更高的准确率。

![2019-02-21-09-36-11](http://www.xdpie.com/2019-02-21-09-36-11.png)




![2019-02-21-09-36-24](http://www.xdpie.com/2019-02-21-09-36-24.png)