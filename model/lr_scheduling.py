import tensorflow as tf
from tensorflow.python import keras
import numpy as np


class MyCosineDecayLearningRate(keras.callbacks.Callback):
    """
    Keras Callback for the learning rate schedule.
    The decay can be either at every update (every mini-batch)
    or at every epoch.
    It follows the half-cosine decay with restart
    in https://arxiv.org/abs/1608.03983 and add decay to
    the maximum learning rate in each cycle.
    """

    def __init__(self, initial_lr, T_cycle, min_lr, n_batches, n_epochs, update_type='batch'):
        """
        Instantiates a Keras Callback for learning rate scheduling.

        Arguments:
            initial_lr: maximum learning rate of the first cycle.
            T_cycle: period of the cycles.
            min_lr: minimum learning rate in all the cycles.
            n_batches: number of batches in one epoch (rounded up if float).
            n_epochs: number of epochs of the training.
            update_type: either 'batch' or 'epoch'.
                Determines if the learning rate changes at every batch or at every epoch.

        Returns:
            A Keras Callback that changes the learning
            rate according to the schedule.
        """
        super(MyCosineDecayLearningRate, self).__init__()
        self.max_lr_curr = initial_lr
        self.all_lr = []
        self.epoch = None
        self.T_cycle = T_cycle
        self.scaled_Tcurrent = np.linspace(start=0, stop=T_cycle, num=T_cycle)
        self.min_lr = min_lr
        self.max_batch_index = 0
        self.update_type = update_type
        self.n_batches = n_batches
        self.n_cycles = np.ceil((self.n_batches * n_epochs) / T_cycle)
        self.max_lr_schedule = np.linspace(start=initial_lr, stop=self.min_lr * 100, num=self.n_cycles + 1)
        self.cycle = 0

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        if self.update_type == 'epoch':
            scheduled_lr = self.lr_schedule(t=epoch)

            tf.keras.backend.set_value(self.model.optimizer.lr, scheduled_lr)
            # print('\nEpoch %05d: Learning rate is %6.6f.' % (epoch, scheduled_lr))
            self.all_lr.append(scheduled_lr)

    def on_batch_begin(self, batch, logs=None):
        if self.update_type == 'batch':
            t = int(self.epoch * self.n_batches + batch)

            scheduled_lr = self.lr_schedule(t=t)

            tf.keras.backend.set_value(self.model.optimizer.lr, scheduled_lr)
            # print('\nBatch %05d: Learning rate is %6.6f.' % (t, scheduled_lr))
            self.all_lr.append(scheduled_lr)

    def lr_schedule(self, t):
        """
        Computes the value of the learning rate.

        Arguments:
            t: the counter of either the global batch or epoch.

        Returns:
            The learning rate for this batch or epoch.

        """
        indT_current = t % self.T_cycle
        T_current = self.scaled_Tcurrent[indT_current]

        lr = self.min_lr + 0.5 * (self.max_lr_curr - self.min_lr) * (1 + np.cos(T_current * np.pi / self.T_cycle))

        if indT_current == self.T_cycle - 1:
            self.cycle += 1
            self.max_lr_curr = self.max_lr_schedule[self.cycle]

        return lr
