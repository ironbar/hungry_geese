import tensorflow as tf


def masked_mean_squared_error(y_true, y_pred):
    mask = y_true[:, :, 1]
    y_true = y_true[:, :, 0]
    squared_difference = tf.square(y_true - y_pred)*mask
    return tf.reduce_sum(squared_difference)/tf.reduce_sum(mask)
