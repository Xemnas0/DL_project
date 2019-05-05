from datetime import datetime
#
from datasets.dataset_loader import load_dataset
from model.randwires import RandWireNN
import tensorflow as tf
from tensorflow.python import keras
from tqdm import tqdm
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
parser.add_argument('--seed', type=int, default=0, help='Random seed initializer.')
parser.add_argument('--graph-mode', type=str, default="WS",
                    help="Random graph family. [ER, WS, BA] (default: WS)")
parser.add_argument('--N', type=int, default=32, help="Number of graph node. (default: 32)")
parser.add_argument('--learning-rate', type=float, default=1e-2, help='Learning rate. (default: --)')
parser.add_argument('--batch-size', type=int, default=32, help='Batch size. (default: --)')
parser.add_argument('--regime', type=str, default="regular",
                    help='[small, regular] (default: regular)')
parser.add_argument('--dataset', type=str, default="MNIST",
                    help='Name of the dataset to use. [CIFAR10, CIFAR100, MNIST] (default: CIFAR10)')
parser.add_argument('--is-train', type=bool, default=True, help="True if training, False if test. (default: True)")
parser.add_argument('--load-model', type=bool, default=False)  # TODO: to load from specified file

args = parser.parse_args()


def loss(model, x, y):
    y_ = model(x)
    # computed_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_, labels=y)
    # lossf = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    # computed_loss = lossf(y, y_)
    computed_loss = keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=y_, from_logits=True)
    # computed_loss = keras.losses.categorical_crossentropy(y_true=y, y_pred=y_, label_smoothing=0)
    # tf.python.losses.softmax_cross_entropy(onehot_labels=y, logits=y_, label_smoothing=0.1)
    return computed_loss


def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
    loss_value = tf.reduce_mean(loss_value)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


def compute_loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    loss = keras.losses.sparse_categorical_crossentropy(y_true=labels, y_pred=logits, from_logits=True)
    return loss


def compute_accuracy(logits, labels):
    predictions = tf.argmax(logits, axis=1)
    print(predictions.numpy())
    return tf.reduce_mean(tf.cast(tf.equal(predictions, labels), tf.float32))


def train_one_step(model, optimizer, x, y):
    with tf.GradientTape(persistent=True) as tape:
        logits = model(x)
        loss = compute_loss(logits, y)

    # compute gradient
    grads = tape.gradient(loss, model.trainable_variables)
    # update to weights
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    accuracy = compute_accuracy(logits, y)

    # loss and accuracy is scalar tensor
    return loss, accuracy


def main():
    (x_train, y_train, Y_train), (x_test, y_test, Y_test) = load_dataset(args.dataset)

    model = RandWireNN(args, input_shape=x_train[0].shape, n_classes=y_train.numpy().max() + 1)

    optimizer = tf.keras.optimizers.Adam(args.learning_rate)
    model.build(input_shape=(None,) + x_train[0].shape)
    model.summary()

    model.compile(optimizer=optimizer, loss=keras.losses.sparse_categorical_crossentropy,
                  metrics=[keras.metrics.sparse_categorical_accuracy])
    #
    # Keras training style
    #

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    # Shuffle and slice the dataset.
    train_dataset = train_dataset.shuffle(buffer_size=x_train.shape[0]).batch(args.batch_size)

    model.fit(train_dataset, epochs=args.epochs)

if __name__ == '__main__':
    main()
