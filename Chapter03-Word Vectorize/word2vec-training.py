from keras import Sequential, Input, Model, optimizers
from keras.layers import Embedding, Flatten, Dense, Dot, Activation, Reshape, Concatenate
from keras.utils import plot_model
from keras_preprocessing import sequence
from keras_preprocessing.sequence import skipgrams
from keras_preprocessing.text import Tokenizer
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

window_size = 5
vector_dim = 100
vocab_size = 10000
epoch = 1


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


if __name__ == '__main__':
    sequences, word_index = collect_data()
    model = create_models()
    model.summary()
    x, y = [], []
    for i, doc in enumerate(sequences):
        sampling_table = sequence.make_sampling_table(vocab_size)
        data, labels = skipgrams(sequence=doc, vocabulary_size=vocab_size, window_size=window_size,negative_samples=5,
                                 sampling_table=sampling_table)
        x.extend(data)
        y.extend(labels)
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
    print('training completed!')
