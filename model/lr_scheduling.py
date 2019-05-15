from tensorflow.python import keras
import numpy as np


class MyLearningRateScheduler(keras.callbacks.Callback):
    """Learning rate scheduler which sets the learning rate according to schedule.

    Arguments:
        schedule: a function that takes an epoch index
            (integer, indexed from 0) and current learning rate
            as inputs and returns a new learning rate as output (float).
    """

    def __init__(self, schedule, initial_lr):
        super(MyLearningRateScheduler, self).__init__()
        self.schedule = schedule
        self.initial_lr = initial_lr
        self.all_lr = []

    # TODO: change to on_batch_begin
    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        # Get the current learning rate from model's optimizer.
        # lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
        # Call schedule function to get the scheduled learning rate.
        scheduled_lr = self.schedule(epoch, self.initial_lr)
        # Set the value back to the optimizer before this epoch starts
        tf.keras.backend.set_value(self.model.optimizer.lr, scheduled_lr)
        print('\nEpoch %05d: Learning rate is %6.6f.' % (epoch, scheduled_lr))
        self.all_lr.append(scheduled_lr)


def lr_schedule(t, initial_lr, T_cycle=20, min_lr=1e-6):
    T_current = t % T_cycle

    lr = min_lr + 0.5 * (initial_lr - min_lr) * (1 + np.cos(T_current * np.pi / T_cycle))

    return lr
