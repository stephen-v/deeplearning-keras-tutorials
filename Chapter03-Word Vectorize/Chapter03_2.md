# word2vec by keras 

word2vec将每个词表示成一个定长的向量，并使得这些向量能较好地表达不同词之间的相似和类比关系。word2vec工具包含了两个模型，即跳字模型（skip-gram） 和连续词袋模型（continuous bag of words，CBOW）。我们重点介绍跳字模型及其训练方法。

## 跳字模型
跳字模型假设基于某个词来生成它在文本序列周围的词。举个例子，假设文本序列是“the”“man”“loves”“his”“son”。以“loves”作为中心词，设背景窗口大小为2。如图10.1所示，跳字模型所关心的是，给定中心词“loves”，生成与它距离不超过2个词的背景词“the”“man”“his”“son”的条件概率，即

$$P(\textrm{"the"},\textrm{"man"},\textrm{"his"},\textrm{"son"}\mid\textrm{``loves"}).$$

假设给定中心词的情况下，背景词的生成是相互独立的，那么上式可以改写成

$$P(\textrm{"the"}\mid\textrm{"loves"})\cdot P(\textrm{"man"}\mid\textrm{"loves"})\cdot P(\textrm{"his"}\mid\textrm{"loves"})\cdot P(\textrm{"son"}\mid\textrm{"loves"}).$$

![2019-02-26-10-06-45](http://www.xdpie.com/2019-02-26-10-06-45.png)

在跳字模型中，每个词被表示成两个$d$维向量，用来计算条件概率。假设这个词在词典中索引为$i$，当它为中心词时向量表示为$\boldsymbol{v}_i\in\mathbb{R}^d$，而为背景词时向量表示为$\boldsymbol{u}_i\in\mathbb{R}^d$。设中心词$w_c$在词典中索引为$c$，背景词$w_o$在词典中索引为$o$，给定中心词生成背景词的条件概率可以通过对向量内积做softmax运算而得到：

$$P(w_o \mid w_c) = \frac{\text{exp}(\boldsymbol{u}_o^\top \boldsymbol{v}c)}{ \sum{i \in \mathcal{V}} \text{exp}(\boldsymbol{u}_i^\top \boldsymbol{v}_c)},$$

其中词典索引集$\mathcal{V} = {0, 1, \ldots, |\mathcal{V}|-1}$。假设给定一个长度为$T$的文本序列，设时间步$t$的词为$w^{(t)}$。假设给定中心词的情况下背景词的生成相互独立，当背景窗口大小为$m$时，跳字模型的似然函数即给定任一中心词生成所有背景词的概率

$$ \prod_{t=1}^{T} \prod_{-m \leq j \leq m,\ j \neq 0} P(w^{(t+j)} \mid w^{(t)}),$$

这里小于1和大于$T$的时间步可以忽略。

## 训练跳字模型(skip-gram)

跳字模型的参数是每个词所对应的中心词向量和背景词向量。训练中我们通过最大化似然函数来学习模型参数，即最大似然估计。这等价于最小化以下损失函数：

$$ - \sum_{t=1}^{T} \sum_{-m \leq j \leq m,\ j \neq 0} \text{log}, P(w^{(t+j)} \mid w^{(t)}).$$

如果使用随机梯度下降，那么在每一次迭代里我们随机采样一个较短的子序列来计算有关该子序列的损失，然后计算梯度来更新模型参数。梯度计算的关键是条件概率的对数有关中心词向量和背景词向量的梯度。根据定义，首先看到

$$\log P(w_o \mid w_c) = \boldsymbol{u}_o^\top \boldsymbol{v}c - \log\left(\sum{i \in \mathcal{V}} \text{exp}(\boldsymbol{u}_i^\top \boldsymbol{v}_c)\right)$$

通过微分，我们可以得到上式中$\boldsymbol{v}_c$的梯度

$$ \begin{aligned} \frac{\partial \text{log}, P(w_o \mid w_c)}{\partial \boldsymbol{v}_c} &= \boldsymbol{u}o - \frac{\sum{j \in \mathcal{V}} \exp(\boldsymbol{u}_j^\top \boldsymbol{v}_c)\boldsymbol{u}j}{\sum{i \in \mathcal{V}} \exp(\boldsymbol{u}_i^\top \boldsymbol{v}_c)}\ &= \boldsymbol{u}o - \sum{j \in \mathcal{V}} \left(\frac{\text{exp}(\boldsymbol{u}_j^\top \boldsymbol{v}c)}{ \sum{i \in \mathcal{V}} \text{exp}(\boldsymbol{u}_i^\top \boldsymbol{v}_c)}\right) \boldsymbol{u}_j\ &= \boldsymbol{u}o - \sum{j \in \mathcal{V}} P(w_j \mid w_c) \boldsymbol{u}_j. \end{aligned} $$

它的计算需要词典中所有词以$w_c$为中心词的条件概率。有关其他词向量的梯度同理可得。

训练结束后，对于词典中的任一索引为$i$的词，我们均得到该词作为中心词和背景词的两组词向量$\boldsymbol{v}_i$和$\boldsymbol{u}_i$。在自然语言处理应用中，一般使用跳字模型的中心词向量作为词的表征向量。

## 近似训练
跳字模型的核心在于使用softmax运算得到给定中心词$w_c$来生成背景词$w_o$的条件概率

$$P(w_o \mid w_c) = \frac{\text{exp}(\boldsymbol{u}_o^\top \boldsymbol{v}c)}{ \sum{i \in \mathcal{V}} \text{exp}(\boldsymbol{u}_i^\top \boldsymbol{v}_c)}.$$

该条件概率相应的对数损失

$$-\log P(w_o \mid w_c) = -\boldsymbol{u}_o^\top \boldsymbol{v}c + \log\left(\sum{i \in \mathcal{V}} \text{exp}(\boldsymbol{u}_i^\top \boldsymbol{v}_c)\right).$$

由于softmax运算考虑了背景词可能是词典$\mathcal{V}$中的任一词，以上损失包含了词典大小数目的项的累加。在上一节中我们看到，不论是跳字模型还是连续词袋模型，由于条件概率使用了softmax运算，每一步的梯度计算都包含词典大小数目的项的累加。对于含几十万或上百万词的较大词典，每次的梯度计算开销可能过大。为了降低该计算复杂度，

## 负采样 
负采样修改了原来的目标函数。给定中心词$w_c$的一个背景窗口，我们把背景词$w_o$出现在该背景窗口看作一个事件，并将该事件的概率计算为

$$P(D=1\mid w_c, w_o) = \sigma(\boldsymbol{u}_o^\top \boldsymbol{v}_c),$$

其中的$\sigma$函数与sigmoid激活函数的定义相同：

$$\sigma(x) = \frac{1}{1+\exp(-x)}.$$

我们先考虑最大化文本序列中所有该事件的联合概率来训练词向量。具体来说，给定一个长度为$T$的文本序列，设时间步$t$的词为$w^{(t)}$且背景窗口大小为$m$，考虑最大化联合概率

$$ \prod_{t=1}^{T} \prod_{-m \leq j \leq m,\ j \neq 0} P(D=1\mid w^{(t)}, w^{(t+j)}).$$

然而，以上模型中包含的事件仅考虑了正类样本。这导致当所有词向量相等且值为无穷大时，以上的联合概率才被最大化为1。很明显，这样的词向量毫无意义。负采样通过采样并添加负类样本使目标函数更有意义。设背景词$w_o$出现在中心词$w_c$的一个背景窗口为事件$P$，我们根据分布$P(w)$采样$K$个未出现在该背景窗口中的词，即噪声词。设噪声词$w_k$（$k=1, \ldots, K$）不出现在中心词$w_c$的该背景窗口为事件$N_k$。假设同时含有正类样本和负类样本的事件$P, N_1, \ldots, N_K$相互独立，负采样将以上需要最大化的仅考虑正类样本的联合概率改写为

$$ \prod_{t=1}^{T} \prod_{-m \leq j \leq m,\ j \neq 0} P(w^{(t+j)} \mid w^{(t)}),$$

其中条件概率被近似表示为 $$ P(w^{(t+j)} \mid w^{(t)}) =P(D=1\mid w^{(t)}, w^{(t+j)})\prod_{k=1,\ w_k \sim P(w)}^K P(D=0\mid w^{(t)}, w_k).$$

设文本序列中时间步$t$的词$w^{(t)}$在词典中的索引为$i_t$，噪声词$w_k$在词典中的索引为$h_k$。有关以上条件概率的对数损失为

![2019-02-26-10-33-13](http://www.xdpie.com/2019-02-26-10-33-13.png)

现在，训练中每一步的梯度计算开销不再与词典大小相关，而与$K$线性相关。当$K$取较小的常数时，负采样在每一步的梯度计算开销较小。

## keras的实现

* 1、获取数据
TB（Penn Tree Bank）是一个常用的小型语料库 [1]。它采样自《华尔街日报》的文章，包括训练集、验证集和测试集。我们将在PTB训练集上训练词嵌入模型。该数据集的每一行作为一个句子。句子中的每个词由空格隔开。在数据集里已经对常见字和非常见字做了划分，取出10000字作为常见字其他用`<unk>`标识。

```Python
def collect_data():
    with open('../datasets/ptb.train.txt', 'r') as f:
        lines = f.readlines()
        # st是sentence的缩写
        raw_dataset = [st.split() for st in lines]
    print('sentences: %d' % len(raw_dataset))

    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(raw_dataset)
    sequences = tokenizer.texts_to_sequences(raw_dataset)
    word_index = tokenizer.word_index  # 最常见10000个词，词典

    return sequences, word_index
```

* 2、构架模型
![2019-02-26-10-15-36](http://www.xdpie.com/2019-02-26-10-15-36.png)
模型部分有两个输入分别是中心词和背景词，根据公式两个向量Dot后的结果,根据负采样中损失函数的定义，我们可以直接使用二元交叉熵损失函数BinaryCrossEntropyLoss
```Python
def create_models():
    # inputs
    w_inputs = Input(shape=(1,), dtype='int32')
    w = Embedding(vocab_size, vector_dim)(w_inputs)

    # context
    c_inputs = Input(shape=(1,), dtype='int32')
    c = Embedding(vocab_size, vector_dim)(c_inputs)
    o = Dot(axes=2)([w, c])
    # o = Reshape((1,))(o)
    o = Flatten()(o)
    o = Activation('sigmoid', name='sigmoid')(o)

    model = Model(inputs=[w_inputs, c_inputs], outputs=o)

    model.compile(optimizer=optimizers.RMSprop(lr=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    # plot_model(model, show_layer_names=True, show_shapes=True, to_file='./plot_model.png')
    return model
```

* 3、构建训练测试数据
根据跳字模型的方法构建背景词和负采样，其中负采样默认为1，`make_sampling_table`做了**二次采样**,文本数据中一般会出现一些高频词，如英文中的“the”“a”和“in”。通常来说，在一个背景窗口中，一个词（如“chip”）和较低频词（如“microprocessor”）同时出现比和较高频词（如“the”）同时出现对训练词嵌入模型更有益。因此，训练词嵌入模型时可以对词进行二次采样。 具体来说，数据集中每个被索引词$w_i$将有一定概率被丢弃，该丢弃概率为

$$ P(w_i) = \max\left(1 - \sqrt{\frac{t}{f(w_i)}}, 0\right),$$

其中 $f(w_i)$ 是数据集中词$w_i$的个数与总词数之比，常数$t$是一个超参数（实验中设为$10^{-4}$）。可见，只有当$f(w_i) > t$时，我们才有可能在二次采样中丢弃词$w_i$，并且越高频的词被丢弃的概率越大。
```
 for i, doc in enumerate(sequences):
        sampling_table = sequence.make_sampling_table(vocab_size)
        data, labels = skipgrams(sequence=doc, vocabulary_size=vocab_size, window_size=window_size,
                                 sampling_table=sampling_table)
        x.extend(data)
        y.extend(labels)
```

* 4、训练模型
我们将模型训练的结果存入word2vet.txt 
```Python
 x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
    x_train_input = [np.array(x) for x in zip(*x_train)]
    x_test_input = [np.array(x) for x in zip(*x_test)]
    history = model.fit(x_train_input, y_train, epochs=epoch, batch_size=512, validation_data=[x_test_input, y_test])
    vectors = model.get_weights()[0]
    f = open('../models/word2vec.txt', 'w')
    f.write('{} {}\n'.format(vocab_size - 1, vector_dim))
    for word, i in word_index.items():
        f.write('{} {}\n'.format(word, ' '.join(map(str, list(vectors[i, :])))))
    f.close()
```
* 5、测试模型
```Python
import gensim

if __name__ == '__main__':
    w2v = gensim.models.KeyedVectors.load_word2vec_format('../models/word2vec_200.txt', binary=False)
    print(w2v.most_similar(positive=['scholar']))

```


