from datasets.dataset_loader import load_dataset
from model.randwires import RandWire
import tensorflow as tf

dataset_name = 'mnist'  # in ['mnist', 'cifar10', 'cifar100']


def main():
    (x_train, y_train), (x_test, y_test) = load_dataset(dataset_name)
    # print(x_train.shape)

    C = 50

    model = RandWire(C, input_size=x_train[0].shape, n_classes=y_train.max() + 1)

    optimizer = tf.keras.optimizers.Adam()
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=20)

if __name__ == '__main__':
    main()
