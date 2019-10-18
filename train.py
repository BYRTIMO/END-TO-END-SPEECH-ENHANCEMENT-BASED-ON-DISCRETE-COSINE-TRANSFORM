# -*- coding: utf-8 -*-

# Imports
import os, sys, time
import numpy as np
import tensorflow as tf
import pickle
import math

import config as cfg
from model import end_to_end
from util import cosine_distance
import data_prepare as dp
# Get training data and validation data sets
train_set = os.path.join(cfg.workdir, "packs/train_set")           
train_speech_names = [os.path.join(train_set, na) for na in os.listdir(train_set) if na.lower().endswith(".p")]
val_set = os.path.join(cfg.workdir, "packs/val_set")             #
val_speech_names = [os.path.join(val_set, na) for na in os.listdir(val_set) if na.lower().endswith(".p")]

def read_pickle(file_path):
  clean, noisy = pickle.load(open(file_path.decode(), 'rb'))
  return clean, noisy

def func(file_path):
  clean, noisy = tf.py_func(read_pickle, [file_path], [tf.float32, tf.float32])
  return clean, noisy

def train(d):
  """train"""
  trainSpeechNames = tf.placeholder(tf.string, shape=[None], name="train_speech_names")
  batch_size = d.batch_size
  height = d.selection
  # TRAINING OPTIMIZER
  global_step = tf.Variable(0, trainable=False, name='global_step')
  lr = tf.Variable(d.lr, trainable=False)
  opt=tf.train.AdamOptimizer(learning_rate=lr, beta1=0., beta2=0.999, epsilon=1e-8)
  # 
  log_file = open(os.path.join(d.workdir, "logfile.txt"), 'w+')
  log_device_file = open(os.path.join(d.workdir, "devicefile.log"), 'w+')
  # Model save path
  model_path = os.path.join(d.workdir, "models")
  dp.create_folder(model_path)
  # initialize dataset
  with tf.name_scope('dataset'):
    dataset = tf.data.Dataset.from_tensor_slices(trainSpeechNames) \
                           .map(func).batch(16)
    iterator = dataset.make_initializable_iterator()
    Ref, Input = iterator.get_next()
  Output = end_to_end(Input, True, d)
  # cosine distance
  half_in = tf.reshape(Input, [-1, height // 16])
  half_clean = tf.reshape(Ref, [-1, height // 16])
  half_out = tf.reshape(Output, [-1, height // 16])
  loss_fn = cosine_distance(half_clean, half_out, half_in)
  grads = opt.compute_gradients(loss_fn, var_list=[var for var in tf.trainable_variables()])
  optimizer_op = opt.apply_gradients(grads, global_step=global_step)
  tf.summary.scalar('loss', loss_fn)
  merged = tf.summary.merge_all()
  # INITIALIZE GPU CONFIG
  config=tf.ConfigProto()
  config.gpu_options.allow_growth=True
  #config.log_device_placement = True
  sess=tf.Session(config=config)
  train_writer = tf.summary.FileWriter(os.path.join(d.workdir, "log/train"), sess.graph)
  sess.run(tf.global_variables_initializer())
  sess.run(iterator.initializer, feed_dict={trainSpeechNames: train_speech_names})
  # Model save
  saver = tf.train.Saver(max_to_keep=25)
  #saver.restore(sess, os.path.join(d.workdir, "models/se_model16_15000.ckpt"))
  #sess.run(tf.assign(lr, d.lr))

  loss_train = np.zeros(10000)
  train_batchs = len(train_speech_names) // batch_size   # Training set batch number
  val_batchs = math.ceil(len(val_speech_names) / batch_size)       # Verification set batch number
  loss_val = np.zeros(val_batchs)
  
  while True:    
    # TRAINING ITERATION
    try:
      summary, _, loss_vec, gs = sess.run([merged, optimizer_op, loss_fn, global_step])
    except tf.errors.OutOfRangeError:
      np.random.seed()
      np.random.shuffle(train_speech_names)
      sess.run(iterator.initializer, feed_dict={trainSpeechNames: train_speech_names})
      continue
    loss_train[gs % 5000] = loss_vec
    
    if gs % 50 == 0:
      train_writer.add_summary(summary, gs)
    if gs % 5000 == 0:
      val_dataset = tf.data.Dataset.from_tensor_slices(val_speech_names) \
                                   .map(func).batch(16)
      val_iteator = val_dataset.make_one_shot_iterator()
      val_Ref, val_Input = val_iteator.get_next()
      val_Output = end_to_end(val_Input, False, d)
      val_half_in = tf.reshape(val_Input, [-1, height // 16])
      val_half_clean = tf.reshape(val_Ref, [-1, height // 16])
      val_half_out = tf.reshape(val_Output, [-1, height // 16])
      # cosine distance
      val_loss_fn = cosine_distance(val_half_clean, val_half_out, val_half_in)
      for i in range(0, val_batchs):
        val_loss = sess.run(val_loss_fn)
        loss_val[i] = val_loss
      val_loss_mean = np.mean(loss_val)
      mean_loss_train = np.mean(loss_train[: 5000])
      print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+
            "\tbatch: %d\ttrain loss: %.4f\tvalidation loss: %.4f\n" % 
            (gs, mean_loss_train, val_loss_mean))
      log_file.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+
                     "\tbatch: %d\ttrain loss: %.4f\tvalidation loss: %.4f\n" % 
                     (gs, mean_loss_train, val_loss_mean))
      log_file.flush()
      saver.save(sess, os.path.join(model_path, "se_model%d.ckpt" % gs))
    if gs == 200000:
     break
  log_file.close()
      
if __name__ == "__main__":
  train(cfg)














