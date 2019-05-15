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

from model.lr_scheduling import MyCosineDecayLearningRate
from model.resnets import ResNet
import pickle
import matplotlib.pyplot as plt
from utils import plot

parser = argparse.ArgumentParser('parameters')

parser.add_argument('--epochs', type=int, default=200, help='Number of epochs. (default: 100)')
parser.add_argument('--P', type=float, default=0.75, help='Graph edge probability. (default: 0.75)')
parser.add_argument('--C', type=int, default=8,
                    help='Number of channels. (default: --)')
parser.add_argument('--K', type=int, default=4,
                    help='Each node is connected to k nearest neighbors in ring topology. (default: 4)')
parser.add_argument('--M', type=int, default=1,
                    help='Number of edges to attach from a new node to existing nodes. (default: 5)')
parser.add_argument('--seed', type=int, default=0, help='Random seed initializer.')
parser.add_argument('--graph-mode', type=str, default="WS",
                    help="Random graph family. [ER, WS, BA] (default: WS)")
parser.add_argument('--N', type=int, default=8, help="Number of graph node. (default: 32)")
parser.add_argument('--stages', type=int, default=1, help='Number of random layers. (default: 1)')
parser.add_argument('--learning-rate', type=float, default=1e-2, help='Learning rate. (default: --)')
parser.add_argument('--batch-size', type=int, default=128, help='Batch size. (default: --)')
parser.add_argument('--regime', type=str, default="small",
                    help='[small, regular] (default: regular)')
parser.add_argument('--dataset', type=str, default="MNIST",
                    help='Name of the dataset to use. [CIFAR10, CIFAR100, MNIST, TINY_IMAGENET] (default: CIFAR10)')
parser.add_argument('--distributed', type=bool, default=False)
parser.add_argument('--augmented', type=bool, default=True)
parser.add_argument('--stride', type=int, default=2)

args = parser.parse_args()

np.random.seed(args.seed)

early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5,
                                               verbose=0, mode='auto', restore_best_weights=True)


def main():
    (x_train, y_train), (x_test, y_test) = load_dataset(args.dataset)

    lr_decay = MyCosineDecayLearningRate(initial_lr=args.learning_rate, T_cycle=150, min_lr=1e-5, update_type='batch',
                                         n_batches=np.ceil(x_train.shape[0] / args.batch_size))

    callbacks = [lr_decay]

    if not args.distributed:
        model = RandWireNN(args, input_shape=x_train[0].shape, n_classes=y_train.max() + 1)
        # model = ResNet(args, input_shape=x_train[0].shape, n_classes=y_train.max() + 1).get_ready_model()

        # model = applications.vgg16.VGG16(weights=None, include_top=True, input_shape=x_train[0].shape)

        # optimizer = keras.optimizers.Adam(args.learning_rate, decay=args.decay)
        optimizer = keras.optimizers.SGD(lr=args.learning_rate, momentum=0.9, nesterov=True)

        model.build(input_shape=(None,) + x_train[0].shape)
        model.summary()

        model.compile(optimizer=optimizer, loss=keras.losses.sparse_categorical_crossentropy,
                      metrics=[keras.metrics.sparse_categorical_accuracy])

        # model.save_graph_image(path='./graph_images/')

        if args.augmented:
            print('In augmented')
            n_val = x_train.shape[0] // 10
            x_val = x_train[:n_val]
            x_train = x_train[n_val:]
            y_val = y_train[:n_val]
            y_train = y_train[n_val:]

            image_gen = ImageDataGenerator(rotation_range=15,
                                           width_shift_range=0.1,
                                           height_shift_range=0.1,
                                           shear_range=0.01,
                                           zoom_range=[0.9, 1.25],
                                           horizontal_flip=True,
                                           vertical_flip=False,
                                           fill_mode='reflect',
                                           data_format='channels_last',
                                           brightness_range=[0.5, 1.5])

            history = model.fit_generator(image_gen.flow(x_train, y_train, batch_size=args.batch_size),
                                          steps_per_epoch=np.ceil(x_train.shape[0] / args.batch_size),
                                          epochs=args.epochs,
                                          validation_data=(x_val, y_val), callbacks=callbacks)
        else:
            # history = model.fit(x_train, y_train, epochs=args.epochs, validation_split=0.1)
            history = model.fit(x_train, y_train, epochs=args.epochs, validation_split=0.1, callbacks=callbacks)

        loss, acc = model.evaluate(x_test, y_test)
    else:
        mirrored_strategy = tf.distribute.MirroredStrategy()
        with mirrored_strategy.scope():
            model = RandWireNN(args, input_shape=x_train[0].shape, n_classes=y_train.max() + 1)
            # model = ResNet(args, input_shape=x_train[0].shape, n_classes=y_train.max() + 1).get_ready_model()

            # model = applications.vgg16.VGG16(weights=None, include_top=True, input_shape=x_train[0].shape)

            optimizer = keras.optimizers.Adam(args.learning_rate)

            model.build(input_shape=(None,) + x_train[0].shape)
            model.summary()

            model.compile(optimizer=optimizer, loss=keras.losses.sparse_categorical_crossentropy,
                          metrics=[keras.metrics.sparse_categorical_accuracy])

            # model.save_graph_image(path='./graph_images/')
            history = model.fit(x_train, y_train, epochs=args.epochs, validation_split=0.1)

            loss, acc = model.evaluate(x_test, y_test)

    results = history.history

    filename = 'history_epochs{4}_{0}_batchsize{1}_eta{2}_{3}'.format(args.dataset,
                                                                      args.batch_size,
                                                                      str(args.learning_rate).replace('.', '_'),
                                                                      model.get_filename(),
                                                                      args.epochs)

    results['test_loss'] = loss
    results['test_acc'] = acc

    plot(results['loss'], results['val_loss'], metric='loss', title='Loss on train and validation data',
         save_name=filename + "loss.png")

    plot(results['sparse_categorical_accuracy'],
         results['val_sparse_categorical_accuracy'],
         metric='accuracy', title='Accuracy on train and validation data', save_name=filename + "acc.png")

    print('test loss is {} and acc is {}'.format(loss, acc))

    plt.plot(lr_decay.all_lr)
    plt.title('Learning rate')
    plt.show()

    pickle_out = open(filename + ".pickle", "wb")
    pickle.dump(results, pickle_out)
    pickle_out.close()


if __name__ == '__main__':
    main()
