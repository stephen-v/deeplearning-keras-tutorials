from keras import models, optimizers, regularizers
from keras.datasets import imdb
from keras.preprocessing.text import Tokenizer
import numpy as np
from keras import layers
from keras.utils.np_utils import to_categorical
from keras.layers import Dense
import matplotlib.pyplot as plt

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
