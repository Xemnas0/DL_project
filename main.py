from datasets.dataset_loader import load_dataset
from model.randwires import RandWireNN
import tensorflow as tf
import argparse

parser = argparse.ArgumentParser('parameters')

parser.add_argument('--epochs', type=int, default=100, help='Number of epochs. (default: 100)')
parser.add_argument('--P', type=float, default=0.75, help='Graph edge probability. (default: 0.75)')
parser.add_argument('--C', type=int, default=20,
                    help='Number of channels. (default: --)')
parser.add_argument('--K', type=int, default=4,
                    help='Each node is connected to k nearest neighbors in ring topology. (default: 4)')
parser.add_argument('--M', type=int, default=5,
                    help='Number of edges to attach from a new node to existing nodes. (default: 5)')
parser.add_argument('--graph-mode', type=str, default="WS",
                    help="Random graph family. [ER, WS, BA] (default: WS)")
parser.add_argument('--N', type=int, default=32, help="Number of graph node. (default: 32)")
parser.add_argument('--learning-rate', type=float, default=1e-2, help='Learning rate. (default: --)')
parser.add_argument('--batch-size', type=int, default=32, help='Batch size. (default: --)')
parser.add_argument('--regime', type=str, default="regular",
                    help='[small, regular] (default: regular)')
parser.add_argument('--dataset', type=str, default="CIFAR10",
                    help='Name of the dataset to use. [CIFAR10, CIFAR100, MNIST] (default: CIFAR10)')
parser.add_argument('--is-train', type=bool, default=True, help="True if training, False if test. (default: True)")
parser.add_argument('--load-model', type=bool, default=False) # TODO: to load from specified file

args = parser.parse_args()


def main():
    (x_train, y_train), (x_test, y_test) = load_dataset(args.dataset)

    model = RandWireNN(args, input_shape=x_train[0].shape, n_classes=y_train.max() + 1)

    optimizer = tf.keras.optimizers.Adam(args.learning_rate)
    model.build(input_shape=(args.batch_size,) + x_train[0].shape)
    model.summary()
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=args.epochs, batch_size=args.batch_size)


if __name__ == '__main__':
    main()
