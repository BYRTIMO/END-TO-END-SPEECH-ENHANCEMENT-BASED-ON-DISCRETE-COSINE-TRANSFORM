# END-TO-END-SPEECH-ENHANCEMENT-BASED-ON-DISCRETE-COSINE-TRANSFORM
Reference papaer: END-TO-END SPEECHENHANCEMENT BASED ON DISCRETE COSINE TRANSFORM 

arXiv: https://arxiv.org/abs/1910.07840

Envirments setup:

  1)Tensorflow 1.13

  2)librosa, numpy, scipy

Usage:

  1) Run data_prepare.py to create packs feature, command line:

python data_prepare.py pack_waves --workspace=.. --clean_dir=path_to_clean_wav --noisy_dir=path_to_noisy_wav

  2) Run train.py to train the U-net, command line:

python train.py
