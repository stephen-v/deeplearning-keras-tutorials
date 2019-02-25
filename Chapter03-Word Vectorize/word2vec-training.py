from keras import Sequential, Input, Model
from keras.layers import Embedding, Flatten, Dense, Dot, Activation, Reshape
from keras_preprocessing import sequence
from keras_preprocessing.sequence import skipgrams
from keras_preprocessing.text import Tokenizer
import numpy as np

window_size = 5
vector_dim = 20
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
    o = Reshape((1,))(o)
    o = Activation('sigmoid')(o)

    model = Model(inputs=[w_inputs, c_inputs], outputs=o)
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model


if __name__ == '__main__':
    sequences, word_index = collect_data()
    model = create_models()
    for epoch_i in range(epoch):
        loss = 0.
        for i, doc in enumerate(sequences):
            sampling_table = sequence.make_sampling_table(vocab_size)
            data, labels = skipgrams(sequence=doc, vocabulary_size=vocab_size, window_size=5,
                                     sampling_table=sampling_table)
            x = [np.array(x) for x in zip(*data)]
            y = np.array(labels, dtype=np.int32)
            if x:
                loss += model.train_on_batch(x, y)
        print('epoch:%d , loss：%d' % (epoch_i, loss))
    model.save_weights('../models/word2vec.model')
    print('training completed!')
