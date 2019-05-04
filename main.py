from datetime import datetime

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
    y = tf.dtypes.cast(y, tf.int32)
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
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


def compute_loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    return tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels))


def compute_accuracy(logits, labels):
    labels = tf.cast(labels, tf.int64)
    predictions = tf.argmax(logits, axis=1)
    return tf.reduce_mean(tf.cast(tf.equal(predictions, labels), tf.float32))


def train_one_step(model, optimizer, x, y):
    with tf.GradientTape() as tape:
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

    model = RandWireNN(args, input_shape=x_train[0].shape, n_classes=y_train.max() + 1)

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
    # model.fit(train_dataset, epochs=args.epochs)

    #
    # # TF2.0 training style
    #
    # train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train, Y_train))
    # # Shuffle and slice the dataset.
    # train_dataset = train_dataset.shuffle(buffer_size=x_train.shape[0]).batch(args.batch_size)

    # Now we get a test dataset.
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_dataset = test_dataset.batch(64)

    for epoch in range(args.epochs):
        epoch_loss_avg = keras.metrics.Mean()
        epoch_accuracy = keras.metrics.SparseCategoricalAccuracy()

        for x, y in train_dataset:
            # loss, accuracy = train_one_step(model, optimizer, x, y)
            # # Optimize the model
            loss_value, grads = grad(model, x, y)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # Track progress
            epoch_loss_avg(loss_value)  # add current batch loss
            # compare predicted label to actual label
            # epoch_accuracy(tf.argmax(model(x), axis=1, output_type=tf.int32), y)
            epoch_accuracy(y, model(x))
            print(f'Loss: {epoch_loss_avg.result():.4f}\tAcc: {epoch_accuracy.result()*100:.4f}%')
            # print(f'Loss: {loss:.4f}\tAcc: {accuracy*100:.4f}%')
            # end epoch

        train_loss_results.append(epoch_loss_avg.result())
        train_accuracy_results.append(epoch_accuracy.result())

        print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                    epoch_loss_avg.result(),
                                                                    epoch_accuracy.result()))


if __name__ == '__main__':
    main()
