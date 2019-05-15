import tensorflow as tf
from tensorflow.python import keras
import numpy as np


class MyCosineDecayLearningRate(keras.callbacks.Callback):
    """Learning rate scheduler which sets the learning rate according to schedule.

    Arguments:
        schedule: a function that takes an epoch index
            (integer, indexed from 0) and current learning rate
            as inputs and returns a new learning rate as output (float).
    """

    def __init__(self, initial_lr, T_cycle, min_lr, n_batches, n_epochs, update_type='batch'):
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
        self.max_lr_schedule = np.linspace(start=initial_lr, stop=self.min_lr*100, num=self.n_cycles+1)
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
        indT_current = t % self.T_cycle
        T_current = self.scaled_Tcurrent[indT_current]

        lr = self.min_lr + 0.5 * (self.max_lr_curr - self.min_lr) * (1 + np.cos(T_current * np.pi / self.T_cycle))

        if indT_current == self.T_cycle - 1:
            self.cycle += 1
            self.max_lr_curr = self.max_lr_schedule[self.cycle]

        return lr
