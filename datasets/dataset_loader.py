import os

import tensorflow_datasets as tfds
from tensorflow.python import keras
from tensorflow.python.keras.datasets.cifar import load_batch
import tensorflow as tf
import numpy as np

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.datasets.cifar import load_batch

from datasets.load_tiny_imagenet import *

EXISTING_DATASETS = ['MNIST', 'CIFAR10', 'CIFAR100', 'TINY_IMAGENET']


def load_dataset(dataset_name):
    """
    Load the whole dataset in the RAM memory.
    If this is run for the first time, then the
    dataset will be downloaded in ~/.keras/datasets.
    The data have shape (samples, height, width, channels).
    MNIST has channels=1.

    Arguments:
        dataset_name: can be MNIST, CIFAR10, CIFAR100 or TINY_IMAGENET.

    Returns:
        Two tuples representing the training set and the test set
            as (x_train, y_train), (x_test, y_test)
    """

    assert dataset_name in EXISTING_DATASETS, 'Dataset name is not valid. Valid datasets: {}'.format(EXISTING_DATASETS)

    if dataset_name == 'MNIST':
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        x_train = x_train[..., np.newaxis]
        x_test = x_test[..., np.newaxis]
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
    elif dataset_name == 'CIFAR10':
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
    elif dataset_name == 'CIFAR100':
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
    elif dataset_name == 'TINY_IMAGENET':
        save_file = 'datasets/tiny_imagenet_data.npz'

        if not os.path.isfile(save_file):
            path = download_tiny_imagenet()
            [x_train, y_train, x_test, y_test] = load_tiny_imagenet(path)
        else:
            [x_train, y_train, x_test, y_test] = load_from_file(save_file)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')

    print('Dataset {} loaded '.format(dataset_name))

    return (x_train, y_train), (x_test, y_test)


if __name__ == '__main__':
    print(tfds.list_builders())
