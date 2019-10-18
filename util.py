# -*- coding: utf-8 -*-
"""
The creation of this script is mainly to write various loss functions
and explore the effectiveness of the loss function on model training.
"""
import tensorflow as tf
  
def cosine_distance(clean, enhanced, noisy, name='cosine_distance'):
  """
  compute cosine distance between clean and enhanced
  smaller is better
  Ref: "PHASE-AWARE SPEECH ENHANCEMENT WITH DEEP COMPLEX U-NET"
  params:
    clean: tensor, shape=(batch, samples)
    enhanced: tensor, shape=(batch, samples)
    noisy: tensor, hsape=(batch, samples)
  return:
    loss: cosine distance
  """
  with tf.name_scope(name):
    # Calculation noise
    noise = noisy - clean
    # Calculated noise estimate
    noise_n = noisy - enhanced
    # l2 norm
    clean_norm = tf.math.l2_normalize(clean, axis=-1, name='clean_norm')
    enhanced_norm = tf.math.l2_normalize(enhanced, axis=-1, name='enhanced_norm')
    noise_norm = tf.math.l2_normalize(noise, axis=-1, name='noise_norm')
    noise_n_norm = tf.math.l2_normalize(noise_n, axis=-1, name='noise_n_norm')
    # compute energy
    clean_norm_val = tf.maximum(tf.reduce_sum(tf.square(clean), 
                                              axis=-1), 
                                1e-12, 
                                name='clean_norm_val')
    noise_norm_val = tf.maximum(tf.reduce_sum(tf.square(noise), 
                                              axis=-1), 
                                1e-12,
                                name='noise_norm_val')
    # alpha=c_e / (c_e +n_e)
    alpha = clean_norm_val / (clean_norm_val + noise_norm_val)
    # wSDR = -(alpha *sum(c_n * e_n, axis=-1) +(1-alpha) *sum(n_n *nn_n, axis=-1))
    w_angle = alpha * tf.reduce_sum(clean_norm * enhanced_norm, axis=-1) + \
              (1 - alpha) * tf.reduce_sum(noise_norm * noise_n_norm, axis=-1)
    # 
    mean_loss = -tf.reduce_mean(w_angle)
  return mean_loss

if __name__ == '__main__':
  print("Hellow World!!!")
