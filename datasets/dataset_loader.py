import tensorflow_datasets as tfds
from tensorflow.python import keras


def load_dataset(dataset_name):
    existing_datasets = ['mnist', 'cifar10', 'cifar100']
    assert dataset_name in existing_datasets, f'Dataset name is not valid. Valid datasets: {existing_datasets}'

    if dataset_name == 'mnist':
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    elif dataset_name == 'cifar10':
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    elif dataset_name == 'cifar100':
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()

    print(f'Dataset {dataset_name} loaded.')

    return (x_train, y_train), (x_test, y_test)


if __name__ == '__main__':
    print(tfds.list_builders())
