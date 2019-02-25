- [1. 文本向量化](#1-%E6%96%87%E6%9C%AC%E5%90%91%E9%87%8F%E5%8C%96)
  - [1.1. one-hot](#11-one-hot)
  - [1.2. word embedding](#12-word-embedding)
    - [1.2.1. 利用keras embedding 层学习词嵌入](#121-%E5%88%A9%E7%94%A8keras-embedding-%E5%B1%82%E5%AD%A6%E4%B9%A0%E8%AF%8D%E5%B5%8C%E5%85%A5)
    - [1.2.2. 预训练词嵌入](#122-%E9%A2%84%E8%AE%AD%E7%BB%83%E8%AF%8D%E5%B5%8C%E5%85%A5)


# 1. 文本向量化
与其他所有神经网络一样，深度学习模型不会接收原始文本作为输入，它只能处理数值张量。
文本向量化（vectorize）是指将文本转换为数值张量的过程。它有多种实现方法,将文本分解而成的单元（单词、字符或 n-gram）叫作标记（token），将文本分解成标记的过程叫作分词（tokenization）。所有文本向量化过程都是应用某种分词方案，然后将数值向量与生成的标记相关联。这些向量组合成序列张量，被输入到深度神经网络中。将向量与标记相关联的方法有很多种。本节将介绍两种主要方法：对标记做 one-hot 编码（one-hotencoding）与标记嵌入［token embedding，通常只用于单词，叫作词嵌入（word embedding）］

## 1.1. one-hot
one-hot 编码是将标记转换为向量的最常用、最基本的方法。它将每个单词与一个唯一的整数索引相关联，然后将这个整数索引 i 转换为长度为 N 的二进制向量（N 是词表大小）。

```Python
max_features = 10000
imdb.load_data(num_words=max_features)
```
我们调用keras自己的API来获取imdb数据，参数 num_words=10000 的意思是仅保留训练数据中前 10 000 个最常出现的单词。例如：一个句子 $w_1$ $w_2$ $w_3$ 三个词构成，其中 $w_1$在词典中频次排序2，$w_2$ 频次排序9999, $w_3$ 频次排序9998，则输出此句子one-hot编码为[0,1,0...1,1]向量长度等于10000。

```Python
max_features = 10000
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)


def onehot_vectorize(sequences, dimension=max_features):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


x_train = onehot_vectorize(x_train)
x_test = onehot_vectorize(x_test)

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(max_features,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=optimizers.RMSprop(lr=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=20, batch_size=512,
                    validation_data=(x_test, y_test))
test_loss, test_acc = model.evaluate(x_test, y_test)

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

print('test_acc:', test_acc)
```

最后的结果如下，如你所见，训练损失每轮都在降低，训练精度每轮都在提升。这就是梯度下降优化的预期结果——你想要最小化的量随着每次迭代越来越小。但验证损失和验证精度并非如此：它们似乎在第三轮达到最佳值。模型在训练数据上的表现越来越好，但在前所未见的数据(测试集)上不一定表现得越来越好。准确地说，你看到的是过拟合（overfit）：在第二轮之后，你对训练数据过度优化，最终学到的表示仅针对于训练数据，无法泛化到训练集之外的数据。

![2019-02-22-11-01-08](http://www.xdpie.com/2019-02-22-11-01-08.png)
![2019-02-22-11-01-17](http://www.xdpie.com/2019-02-22-11-01-17.png)


## 1.2. word embedding 
将单词与向量相关联还有另一种常用的强大方法，就是使用密集的词向量（word vector），也叫词嵌入（word embedding）。one-hot 编码得到的向量是二进制的、稀疏的（绝大部分元素都是 0）、维度很高的（维度大小等于词表中的单词个数），而词嵌入是低维的浮点数向量（即密集向量，与稀疏向量相对），参见图 6-2。与 one-hot 编码得到的词向量不同，词嵌入是从数据中学习得到的。常见的词向量维度是 256、512 或 1024（处理非常大的词表时）。与此相对，one-hot 编码的词向量维度通常为 20 000 或更高（对应包含 20 000 个标记的词表）。因此，词向量可以将更多的信息塞入更低的维度中.获取词嵌入有两种方法:
* 在完成主任务（比如文档分类或情感预测）的同时学习词嵌入。在这种情况下，一开始是随机的词向量，然后对这些词向量进行学习，其学习方式与学习神经网络的权重相同
* 在不同于待解决问题的机器学习任务上预计算好词嵌入，然后将其加载到模型中。这些词嵌入叫作预训练词嵌入（pretrained word embedding）

### 1.2.1. 利用keras embedding 层学习词嵌入
要将一个词与一个密集向量相关联，最简单的方法就是随机选择向量。这种方法的问题在于，得到的嵌入空间没有任何结构。例如，accurate 和 exact 两个词的嵌入可能完全不同，尽管它们在大多数句子里都是可以互换的，深度神经网络很难对这种杂乱的、非结构化的嵌入空间进行学习。

IMDB 数据集也内置于 Keras 库。它已经过预处理：评论（单词序列）已经被转换为整数序列，其中每个整数代表字典中的某个单词，参数 num_words=10000 的意思是仅保留训练数据中前 10 000 个最常出现的单词,然后我们对句子进行了统一处理，每个句子只保留100个单词，多余的部分截取，不足的部分填充。

```Python
maxlen = 100
max_features = 10000

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)
```

构建模型添加嵌入层,`Embedding(10000, 64, input_length=maxlen)`第一参数10000表示最多的单词数，64表示嵌入维度。再嵌入层后我们进行一次flatten最后得到是(10000，64*100)的矩阵。

```Python
model.add(Embedding(10000, 64, input_length=maxlen))
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
```

可以看到词嵌入和前面onehot训练的结果差不多，同样出现了严重的过拟合准确率也在85%左右
![2019-02-22-11-11-50](http://www.xdpie.com/2019-02-22-11-11-50.png)
![2019-02-22-11-11-59](http://www.xdpie.com/2019-02-22-11-11-59.png)



### 1.2.2. 预训练词嵌入 

有时可用的训练数据很少，以至于只用手头数据无法学习适合特定任务的词嵌入。那么应
该怎么办？你可以从预计算的嵌入空间中加载嵌入向量（你知道这个嵌入空间是高度结构化的，并且
具有有用的属性，即抓住了语言结构的一般特点），而不是在解决问题的同时学习词嵌入。有许多预计算的词嵌入数据库，你都可以下载并在 Keras 的 Embedding 层中使用。word2vec 就是其中之一。另一个常用的是 GloVe（global vectors for word representation，词表示全局向量），由斯坦福大学的研究人员于 2014 年开发。这种嵌入方法基于对词共现统计矩阵进行因式分解。其开发者已经公开了数百万个英文标记的预计算嵌入，它们都是从维基百科数据和 Common Crawl 数据得到的
我们来看一下如何在 Keras 模型中使用 GloVe 嵌入。同样的方法也适用于 word2vec 嵌入或
其他词嵌入数据库。

**1、下载原始imdb数据**

```Python
def get_imdb():
    imdb_dir = 'D:\\datasets\\aclImdb_v1\\aclImdb'
    train_dir = os.path.join(imdb_dir, 'train')
    test_dir = os.path.join(imdb_dir, 'test')

    labels = []
    texts = []
    for label_type in ['neg', 'pos']:
        for dir in [train_dir, test_dir]:
            dir_name = os.path.join(dir, label_type)
            for fname in os.listdir(dir_name):
                if fname[-4:] == '.txt':
                    f = open(os.path.join(dir_name, fname), encoding='utf-8')
                texts.append(f.read())
                f.close()
                if label_type == 'neg':
                    labels.append(0)
                else:
                    labels.append(1)
    return texts, labels
```

**2、准备数据**
这里我们将把选取0.75的数据作为训练集，0.25的数据作为测试集，同时取出10000个常用词作为词库

```Python
def Preparing_data(texts, labels):
    training_samples = int(len(texts) * 0.75)
    validation_samples = int(len(texts) * 0.25)  # 训练集与测试集划分

    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    word_index = tokenizer.word_index  # 最常见10000个词，词典
    print('Found %s unique tokens.' % len(word_index))

    data = pad_sequences(sequences, maxlen=max_len)  # 截取单词不足100的补足100
    labels = np.asarray(labels)

    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]

    x_train = data[:training_samples]
    y_train = labels[:training_samples]
    x_val = data[training_samples: training_samples + validation_samples]
    y_val = labels[training_samples: training_samples + validation_samples]

    return word_index, x_train, y_train, x_val, y_val
```

**3、加载训练好的glove词向量**

```Python
def load_pretrained_model(word_index):
    GLOVE_DIR = os.path.join('../pretrained-models')
    embeddings_index = {}
    f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'), encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    embedding_matrix = np.zeros((max_words, embedding_dim))

    for word, i in word_index.items():
        if i < max_words:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

    print('Found %s word vectors.' % len(embeddings_index))
    return embedding_matrix
```

**4、构建模型**
我们将生成的词向量矩阵放入Embedding层，并将它的trainable属性设置为false这样我们再训练的时候就不会改变此层的权重
```Python
def create_model(embedding_matrix):
    model = Sequential()
    model.add(Embedding(max_words, embedding_dim, input_length=max_len))
    model.add(Flatten())
    model.add(Dense(16, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.layers[0].set_weights([embedding_matrix])
    model.layers[0].trainable = False
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
    return model
```

从最终的训练结果可以看到，使用预加载的词向量其准确度不如前两种，但是在20轮的训练过程中没有出现明显的过拟合想象，这样来说是不是我们加大Epoch会不会提高准确度？？？


![2019-02-22-11-15-11](http://www.xdpie.com/2019-02-22-11-15-11.png)
![2019-02-22-11-15-34](http://www.xdpie.com/2019-02-22-11-15-34.png)

