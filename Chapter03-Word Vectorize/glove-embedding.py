from keras.models import Sequential
from keras.layers import Flatten, Dense, Embedding, Dropout, regularizers
import matplotlib.pyplot as plt
import os
import numpy as np
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer

max_len = 100  # 每条评论最多截取100个单词
max_words = 20000
embedding_dim = 100  # 词嵌入维度


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


def collect_data(texts, labels):
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


def train():
    texts, labels = get_imdb()
    word_index, x_train, y_train, x_val, y_val = collect_data(texts, labels)
    embedding_matrix = load_pretrained_model(word_index)
    model = create_model(embedding_matrix)
    history = model.fit(x_train, y_train, epochs=20, batch_size=512,
                        validation_data=(x_val, y_val))
    test_loss, test_acc = model.evaluate(x_val, y_val)

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


if __name__ == '__main__':
    train()
