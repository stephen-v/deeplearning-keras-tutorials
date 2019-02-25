from keras.datasets import mnist
from keras import models, regularizers
from keras import layers
from keras.utils import plot_model, to_categorical
import matplotlib.pyplot as plt


def get_model():
    """
    create two layers neural network model
    :return: model
    """
    network = models.Sequential()
    network.add(layers.Dense(512, activation='relu',
                             input_shape=(28 * 28,)))
    network.add(layers.Dense(10, activation='softmax'))

    plot_model(network, show_layer_names=True, show_shapes=True, to_file='./plot_model.png')
    return network


def preprocess():
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = train_images.reshape((60000, 28 * 28))
    train_images = train_images.astype('float32') / 255
    test_images = test_images.reshape((10000, 28 * 28))
    test_images = test_images.astype('float32') / 255

    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)
    return train_images, test_images, train_labels, test_labels


if __name__ == '__main__':
    network = get_model()
    network.compile(optimizer='sgd',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    train_images, test_images, train_labels, test_labels = preprocess()
    history = network.fit(train_images, train_labels, epochs=20, batch_size=512,
                          validation_data=(test_images, test_labels))
    test_loss, test_acc = network.evaluate(test_images, test_labels)

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
