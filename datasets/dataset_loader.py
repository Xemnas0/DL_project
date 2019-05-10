import os

import tensorflow_datasets as tfds
from keras.datasets.cifar import load_batch
from tensorflow.python import keras
import tensorflow as tf
import numpy as np

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.datasets.cifar import load_batch

from datasets.load_tiny_imagenet import *

EXISTING_DATASETS = ['MNIST', 'CIFAR10', 'CIFAR100', 'TINY_IMAGENET']


def load_dataset(dataset_name):

    assert dataset_name in EXISTING_DATASETS, f'Dataset name is not valid. Valid datasets: {EXISTING_DATASETS}'

    y_test = []
    Y_test = []
    x_val = []
    y_val = []
    Y_val = []

    if dataset_name == 'MNIST':
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

        x_train = x_train[..., np.newaxis]
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        y_train = tf.cast(y_train, tf.int64)
        y_test = tf.cast(y_test, tf.int64)
        Y_train = tf.one_hot(y_train, y_train.numpy().max()+1, dtype=tf.int32)
        Y_test = tf.one_hot(y_test, y_test.numpy().max()+1, dtype=tf.int32)
    elif dataset_name == 'CIFAR10':

        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        y_train = tf.cast(y_train, tf.int64)
        y_test = tf.cast(y_test, tf.int64)
        Y_train = tf.one_hot(y_train[:, 0], y_train.numpy().max() + 1, dtype=tf.int32)
        Y_test = tf.one_hot(y_test, y_test.numpy().max()+1, dtype=tf.int32)
    elif dataset_name == 'CIFAR100':
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        y_train = tf.cast(y_train, tf.int64)
        y_test = tf.cast(y_test, tf.int64)
        Y_train = tf.one_hot(y_train[:, 0], y_train.numpy().max() + 1, dtype=tf.int32)
        Y_test = tf.one_hot(y_test, y_test.numpy().max()+1, dtype=tf.int32)

    elif dataset_name == 'TINY_IMAGENET':
        path = 'datasets/tiny-imagenet-200'
        save_file = 'datasets/tiny_imagenet_data.npz'
        if not os.path.isfile(save_file):
            [x_train, y_train, x_val, y_val, x_test] = load_tiny_imagenet(path)
        else:
            [x_train, y_train, x_val, y_val, x_test] = load_from_file(save_file)

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_val = x_val.astype('float32')

        y_train = tf.cast(y_train, tf.int64)
        Y_train = tf.one_hot(y_train[:, 0], y_train.numpy().max() + 1, dtype=tf.int32)

        y_val = tf.cast(y_val, tf.int64)
        Y_val = tf.one_hot(y_val[:, 0], y_val.numpy().max() + 1, dtype=tf.int32)



    print(f'Dataset {dataset_name} loaded.')

    return (x_train, y_train, Y_train), (x_val, y_val, Y_val), (x_test, y_test, Y_test)


if __name__ == '__main__':
    print(tfds.list_builders())
