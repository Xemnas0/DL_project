from datetime import datetime
#
from datasets.dataset_loader import load_dataset
from model.randwires import RandWireNN
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
import argparse
import numpy as np

parser = argparse.ArgumentParser('parameters')

parser.add_argument('--epochs', type=int, default=100, help='Number of epochs. (default: 100)')
parser.add_argument('--P', type=float, default=0.5, help='Graph edge probability. (default: 0.75)')
parser.add_argument('--C', type=int, default=32,
                    help='Number of channels. (default: --)')
parser.add_argument('--K', type=int, default=2,
                    help='Each node is connected to k nearest neighbors in ring topology. (default: 4)')
parser.add_argument('--M', type=int, default=1,
                    help='Number of edges to attach from a new node to existing nodes. (default: 5)')
parser.add_argument('--seed', type=int, default=0, help='Random seed initializer.')
parser.add_argument('--graph-mode', type=str, default="WS",
                    help="Random graph family. [ER, WS, BA] (default: WS)")
parser.add_argument('--N', type=int, default=32, help="Number of graph node. (default: 32)")
parser.add_argument('--stages', type=int, default=3, help='Number of random layers. (default: 1)')
parser.add_argument('--learning-rate', type=float, default=1e-2, help='Learning rate. (default: --)')
parser.add_argument('--batch-size', type=int, default=32, help='Batch size. (default: --)')
parser.add_argument('--regime', type=str, default="small",
                    help='[small, regular] (default: regular)')
parser.add_argument('--dataset', type=str, default="MNIST",
                    help='Name of the dataset to use. [CIFAR10, CIFAR100, MNIST] (default: CIFAR10)')
parser.add_argument('--is-train', type=bool, default=True, help="True if training, False if test. (default: True)")
parser.add_argument('--load-model', type=bool, default=False)  # TODO: to load from specified file
parser.add_argument('--checkpoint_path', type=str, default='training_checkpoint/')
args = parser.parse_args()

np.random.seed(args.seed)


def main():
    (x_train, y_train, Y_train), (x_test, y_test, Y_test) = load_dataset(args.dataset)

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=x_train.shape[0]).batch(args.batch_size)
    # TODO: add validation set
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_dataset = test_dataset.shuffle(buffer_size=y_test.shape[0]).batch(args.batch_size)

    if args.load_model:
        checkpoint_path = args.checkpoint_path + "cp-{epoch:04d}.ckpt"
        checkpoint_dir = os.path.dirname(checkpoint_path)

        latest = tf.train.latest_checkpoint(checkpoint_dir)

        model = RandWireNN(args, input_shape=x_train[0].shape, n_classes=y_train.numpy().max() + 1)

        model.load_weights(latest)

        model.summary()

        optimizer = tf.keras.optimizers.Adam(args.learning_rate)
        model.compile(optimizer=optimizer, loss=keras.losses.sparse_categorical_crossentropy,
                      metrics=[keras.metrics.sparse_categorical_accuracy])

        if args.is_train:

            # Create checkpoint callback
            checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                                     verbose=1,
                                                                     period=10)

            callbacks = [checkpoint_callback]

            model.fit(train_dataset, epochs=args.epochs, callbacks=callbacks)

        else:

            loss, acc = model.evaluate(test_dataset)

            print(f'test loss: {loss:.4f}\ttest acc: {acc:.2f*100}%')

    else:

        model = RandWireNN(args, input_shape=x_train[0].shape, n_classes=y_train.numpy().max() + 1)

        optimizer = tf.keras.optimizers.Adam(args.learning_rate)

        model.build(input_shape=(None,) + x_train[0].shape)
        model.summary()

        model.compile(optimizer=optimizer, loss=keras.losses.sparse_categorical_crossentropy,
                      metrics=[keras.metrics.sparse_categorical_accuracy])

        # train_datagen = make_generator()
        # train_dataset_augmented = train_datagen.flow(x_train, y=y_train, batch_size=args.batch_size)

        # train_dataset = tf.data.Dataset.from_generator(make_generator, (tf.float32, tf.float32))

        checkpoint_path = args.checkpoint_path + "cp-{epoch:04d}.ckpt"

        # Create checkpoint callback
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                                 verbose=1,
                                                                 period=10)

        callbacks = [checkpoint_callback]

        model.fit(train_dataset, epochs=args.epochs, callbacks=callbacks)


if __name__ == '__main__':
    main()
