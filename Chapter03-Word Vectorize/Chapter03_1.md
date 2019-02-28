# 过拟合与欠拟合 

我们来看下上一节中对minist训练，如下图，可以看到模型在验证数据上的性能总是在几轮后达到最高点，然后开始下降。也就是说，模型很快就在训练数据上开始过拟合。如你所见，训练损失每轮都在降低，训练精度每轮都在提升。这就是梯度下降优化的预期结果——你想要最小化的量随着每次迭代越来越小。但验证损失和验证精度并非如此：它们似乎在第三轮达到最佳值。这就是我们之前警告过的一种情况：模型在训练数据上的表现越来越好，但在前所未见的数据上不一定表现得越来越好。准确地说，你看到的是过拟合（overfit）：在第二轮之后，你对训练数据过度优化，最终学到的表示仅针对于训练数据，无法泛化到训练集之外的数据

![2019-02-21-21-50-11](http://www.xdpie.com/2019-02-21-21-50-11.png)
![2019-02-21-21-50-23](http://www.xdpie.com/2019-02-21-21-50-23.png)

为了防止模型从训练数据中学到错误或无关紧要的模式，最优解决方法是获取更多的训练数据。模型的训练数据越多，泛化能力自然也越好。如果无法获取更多数据，次优解决方法是调节模型允许存储的信息量，或对模型允许存储的信息加以约束。如果一个网络只能记住几个模式，那么优化过程会迫使模型集中学习最重要的模式，这样更可能得到良好的泛化。这种降低过拟合的方法叫作正则化（regularization）


### 添加权重正则化
**L2正则化**
L2 正则化的想法是增加⼀个额外的项到代价函数上，这个项叫做正则化项。下⾯是正则化后的的交叉熵
$$
C=-\frac{1}{n} \sum\limits_{x}[yln\hat{y} + (1-y)ln(1-\hat{y})] + \frac{\lambda}{2n}\sum\limits_{w}w^2 \tag*{2.6} 
$$

其中第⼀个项就是常规的交叉熵的表达式。第⼆个现在加⼊的就是所有权重的平⽅的和。然后使⽤⼀个因⼦ $\frac{\lambda}{2n}$ 进⾏量化调整，其中 $λ > 0$ 可以称为正则化参数，⽽ n 就是训练集合的⼤⼩.我们将式(2.6)化简求梯度。

$$
C=C_o + \frac{\lambda}{2n}\sum\limits_{w}w^2 \tag*{2.7} 
$$

特别地，我们需要知道如何计算对⽹络中所有权重和偏置的偏导数 $∂C/∂w$ 和 $∂C/∂b$。
对式(2.7) 进⾏求偏导数得

$$
\frac{\partial{C}}{\partial{w}} = \frac{\partial{C_o}}{\partial{w}} + \frac{\lambda}{n}w \tag*{2.8}
$$ 

$$
\frac{\partial{C}}{\partial{b}} = \frac{\partial{C_o}}{\partial{b}} \tag*{2.9}
$$
可以看到对于偏置项b的学习规则不发生变化，而对于权重项w的学习规则发生变化：

$$
w \to w - \eta \frac{\partial{C_o}}{\partial{w}} - \eta\frac{\lambda}{n}w=(1- \eta\frac{\lambda}{n})w -\eta \frac{\partial{C_o}}{\partial{w}} 
$$

这种调整有时被称为权重衰减，因为它使得权重变小。现在我们再keras中改进下之前的代码加入了L2正则化,可以看见数据测试的准确率是持续增加的。

```Python
network.add(layers.Dense(512,kernel_regularizer=regularizers.l2(),activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10,kernel_regularizer=regularizers.l2(),activation='softmax'))
```


### 添加Dropout正则化
dropout 是神经网络最有效也最常用的正则化方法之一，它是由多伦多大学的 Geoffrey Hinton和他的学生开发的。对某一层使用 dropout，就是在训练过程中随机将该层的一些输出特征舍弃（设置为 0）。假设在训练过程中，某一层对给定输入样本的返回值应该是向量 [0.2, 0.5,1.3, 0.8, 1.1] 。使用 dropout 后，这个向量会有几个随机的元素变成 0，比如 [0, 0.5,1.3, 0, 1.1] 。dropout 比率（dropout rate）是被设为 0 的特征所占的比例，通常在 0.2~0.5范围内。测试时没有单元被舍弃，而该层的输出值需要按 dropout 比率缩小，因为这时比训练时有更多的单元被激活，需要加以平衡。

```Python
network.add(layers.Dense(512,activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dropout(0.5))
network.add(layers.Dense(10,kernel_regularizer=regularizers.l2(),activation='softmax'))
```

由上图可以看到再加入了dropout正则化后，测试的准确率同样也是持续增加的


![2019-02-21-21-52-48](http://www.xdpie.com/2019-02-21-21-52-48.png)
![2019-02-21-21-52-57](http://www.xdpie.com/2019-02-21-21-52-57.png)
