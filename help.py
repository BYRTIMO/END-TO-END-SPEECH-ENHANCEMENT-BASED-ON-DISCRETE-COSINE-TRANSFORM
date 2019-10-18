# -*- coding: utf-8 -*-
"""
20191017
"""
# Imports
import numpy as np
import tensorflow as tf

def frame(signals, frame_length, frame_step, winfunc=tf.signal.hamming_window):
  framed_signals = tf.signal.frame(signals, frame_length, frame_step, pad_end=False)
  if winfunc is not None:
    window = winfunc(frame_length, dtype=tf.float32)
    framed_signals *= window
  return framed_signals

def over_lap_and_add(framed_signals, frame_length, frame_step, winfunc=tf.signal.hamming_window):
  """overlap and add
  params：
    framed_signals: tf.float32, shape=[batch, n_frames, frame_length]
    frame_length: Window length
    frame_step: frame shift
  return：
    signals: tf.float32, shape=[batch, x_length]
  """
  shape = tf.shape(framed_signals)
  n_frames = shape[1]
  # Generate de-overlapping windows
  if winfunc is not None:
    window = winfunc(frame_length, dtype=tf.float32)
    window = tf.reshape(window, [1, frame_length])
    window = tf.tile(window, [n_frames, 1])
    window = tf.signal.overlap_and_add(window, frame_step)
  signals = tf.signal.overlap_and_add(framed_signals, frame_step)
  signals /= window
  signals = tf.cast(signals, tf.float32)
  return signals