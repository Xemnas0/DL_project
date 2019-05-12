from datetime import datetime
import os
from datasets.dataset_loader import load_dataset
from datasets.load_tiny_imagenet import load_tinyimagenet_dict
from model.randwires import RandWireNN
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
import argparse
import numpy as np

from model.resnets import ResNet
import pickle

parser = argparse.ArgumentParser('parameters')

parser.add_argument('--epochs', type=int, default=1, help='Number of epochs. (default: 100)')
parser.add_argument('--P', type=float, default=0.75, help='Graph edge probability. (default: 0.75)')
parser.add_argument('--C', type=int, default=32,
                    help='Number of channels. (default: --)')
parser.add_argument('--K', type=int, default=4,
                    help='Each node is connected to k nearest neighbors in ring topology. (default: 4)')
parser.add_argument('--M', type=int, default=1,
                    help='Number of edges to attach from a new node to existing nodes. (default: 5)')
parser.add_argument('--seed', type=int, default=0, help='Random seed initializer.')
parser.add_argument('--graph-mode', type=str, default="BA",
                    help="Random graph family. [ER, WS, BA] (default: WS)")
parser.add_argument('--N', type=int, default=4, help="Number of graph node. (default: 32)")
parser.add_argument('--stages', type=int, default=1, help='Number of random layers. (default: 1)')
parser.add_argument('--learning-rate', type=float, default=1e-2, help='Learning rate. (default: --)')
parser.add_argument('--batch-size', type=int, default=32, help='Batch size. (default: --)')
parser.add_argument('--regime', type=str, default="small",
                    help='[small, regular] (default: regular)')
parser.add_argument('--dataset', type=str, default="MNIST",
                    help='Name of the dataset to use. [CIFAR10, CIFAR100, MNIST, TINY_IMAGENET] (default: CIFAR10)')
args = parser.parse_args()

np.random.seed(args.seed)


def main():
    (x_train, y_train), (x_test, y_test) = load_dataset(args.dataset)

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(args.batch_size)

    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_dataset = test_dataset.shuffle(buffer_size=y_test.shape[0]).batch(args.batch_size)

    model = RandWireNN(args, input_shape=x_train[0].shape, n_classes=y_train.max() + 1)
    # model = ResNet(args, input_shape=x_train[0].shape, n_classes=y_train.max() + 1).get_ready_model()

    # model = applications.vgg16.VGG16(weights=None, include_top=True, input_shape=x_train[0].shape)

    optimizer = keras.optimizers.Adam(args.learning_rate)

    model.build(input_shape=(None,) + x_train[0].shape)
    model.summary()

    model.compile(optimizer=optimizer, loss=keras.losses.sparse_categorical_crossentropy,
                  metrics=[keras.metrics.sparse_categorical_accuracy])

    model.save_graph_image(path='./graph_images/')

    history = model.fit(train_dataset, epochs=args.epochs)
    results = history.history

    loss, acc = model.evaluate(test_dataset)
    print(f'test loss: {loss:.4f}\ttest acc: {acc:.2f*100}%')

    pickle_out = open("dict.pickle","wb")
    pickle.dump(results, pickle_out)
    pickle_out.close()

if __name__ == '__main__':
    main()
