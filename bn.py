# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
def batch_normalization(Input, axis=-1, momentum=0.99, is_training=True):
  """
  Use the minibatch statistic while training, and update the moving average statistic 
  but not use (training=True), use the sliding average statistic but not update (training=False)

  """
  learning_phase = tf.convert_to_tensor(is_training)
  with tf.variable_scope("batch_norm", reuse=tf.AUTO_REUSE):
    # Get input parameters
    shape = Input.shape
    ndim = len(shape)
    reduction_axes = list(range(ndim))
    del reduction_axes[axis]
    # Define initial weight
    beta = tf.get_variable('beta', shape[axis], dtype=tf.float32, initializer=tf.constant_initializer(0.))
    gamma = tf.get_variable('gamma', shape[axis], dtype=tf.float32, initializer=tf.constant_initializer(1.))
    moving_mean = tf.get_variable('moving_mean', shape[axis], dtype=tf.float32, initializer=tf.constant_initializer(0.), trainable=False)
    moving_var = tf.get_variable('moving_var', shape[axis], dtype=tf.float32, initializer=tf.constant_initializer(1.), trainable=False)
    # summary
    tf.summary.histogram(name='moving_mean', values=moving_mean)
    tf.summary.histogram(name='moving_var', values=moving_var)
    def update_state():
      # Calculate input statistics
      mean, variance = tf.nn.moments(Input, axes=reduction_axes)
      # Defining update nodes
      update_op = [tf.assign_sub(moving_mean, (1 - momentum) * (moving_mean - mean)), 
                   tf.assign_sub(moving_var, (1 - momentum) * (moving_var - variance))]
      with tf.control_dependencies(update_op):
        mean = tf.identity(mean)
        variance = tf.identity(variance)
      return mean, variance
    mu, var = tf.cond(learning_phase, update_state, lambda: (moving_mean, moving_var))
    # batch norm
    out = tf.nn.batch_normalization(Input, mu, var, beta, gamma, 1e-8)
  return out
