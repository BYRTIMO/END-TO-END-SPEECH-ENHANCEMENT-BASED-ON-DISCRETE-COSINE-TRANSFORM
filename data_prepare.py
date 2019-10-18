# -*- coding: utf-8 -*-
"""
2019-04-23
"""
import os, sys
import librosa
import soundfile
import argparse
import numpy as np
import math
import pickle
import time
import config as cfg

def create_folder(fd):
  if not os.path.exists(fd):
    os.makedirs(fd)

def read_audio(path, tar_fs=None):
  (audio, fs) = soundfile.read(path)
  if tar_fs is not None and fs != tar_fs:
    audio = librosa.resample(audio, orig_sr=fs, target_sr=tar_fs)
    fs = tar_fs
  audio = audio.astype(np.float32)
  return audio, fs

def pack_waves(args):
  """
  pack waves
  """
  workspace = args.workspace
  clean_dir = args.clean_dir
  noisy_dir = args.noisy_dir
  slice_len = cfg.selection

  strides = slice_len // 2
  #window = np.hamming(slice_len)

  val_set = 1500                       # Verification set data number

  speech_names = [na.split(".")[0] for na in os.listdir(clean_dir) if na.lower().endswith('.wav')] 
  out_csv_path = os.path.join(workspace, "mixture_csvs")
  create_folder(out_csv_path)
  pickle.dump(speech_names, 
              open(os.path.join(out_csv_path, "train.csv"), 'wb'), 
              protocol=pickle.HIGHEST_PROTOCOL)
  np.random.seed()
  ids = np.random.permutation(len(speech_names))
  t1 = time.time()
  cnt = 0
  index = ids[val_set: ]
  out_path = os.path.join(workspace, "packs", "train_set")
  create_folder(out_path)
  for i in index:
    audio_clean, _ = read_audio(os.path.join(clean_dir, "%s.wav" % speech_names[i]))
    audio_noise, _ = read_audio(os.path.join(noisy_dir, "%s.wav" % speech_names[i]))
    n_samples = audio_clean.shape[0]
    slice_num = math.ceil((n_samples - slice_len) / strides) + 1
    for j in range(slice_num):
      if j * strides + slice_len > n_samples:
        slice_clean = audio_clean[-slice_len: ]
        slice_noise = audio_noise[-slice_len: ]
      else:
        slice_clean = audio_clean[j * strides: j * strides + slice_len]
        #slice_clean *= window
        slice_noise = audio_noise[j * strides: j * strides + slice_len]
        #slice_noise *= window
      data = [slice_clean, slice_noise]
    
      pickle.dump(data, open(os.path.join(out_path, "%s_%d.p" % (speech_names[i], j)), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
      if cnt % 100 == 0:
        print(cnt)
      cnt += 1
  print("packing train set wave time: %s" % (time.time() - t1))
  out_path = os.path.join(workspace, "packs", "val_set")
  create_folder(out_path)
  index = ids[: val_set]
  for i in index:
    audio_clean, _ = read_audio(os.path.join(clean_dir, "%s.wav" % speech_names[i]))
    audio_noise, _ = read_audio(os.path.join(noisy_dir, "%s.wav" % speech_names[i]))
    n_samples = audio_clean.shape[0]
    slice_num = math.ceil((n_samples - slice_len) / strides) + 1
    for j in range(slice_num):
      if j * strides + slice_len > n_samples:
        slice_clean = audio_clean[-slice_len: ]
        slice_noise = audio_noise[-slice_len: ]
      else:
        slice_clean = audio_clean[j * strides: j * strides + slice_len]
        #slice_clean *= window
        slice_noise = audio_noise[j * strides: j * strides + slice_len]
        #slice_noise *= window
      data = [slice_clean, slice_noise]
    
      pickle.dump(data, open(os.path.join(out_path, "%s_%d.p" % (speech_names[i], j)), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
      if cnt % 100 == 0:
        print(cnt)
      cnt += 1
  print("packing total wave time: %s" % (time.time() - t1))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  subparsers = parser.add_subparsers(dest='mode')

  parser_pack_waves = subparsers.add_parser('pack_waves')
  parser_pack_waves.add_argument('--workspace', type=str, required=True)
  parser_pack_waves.add_argument('--clean_dir', type=str, required=True)
  parser_pack_waves.add_argument('--noisy_dir', type=str, required=True)

  args = parser.parse_args()
  if args.mode == 'pack_waves':
    pack_waves(args)
  else:
    raise Exception("Error!")
