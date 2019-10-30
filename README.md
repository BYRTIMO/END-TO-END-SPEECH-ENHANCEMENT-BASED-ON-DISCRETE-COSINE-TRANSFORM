# END-TO-END-SPEECH-ENHANCEMENT-BASED-ON-DISCRETE-COSINE-TRANSFORM
papaer path:https://arxiv.org/abs/1910.07840

First run data_prepare.py to create packs feature, command line:

python data_prepare.py pack_waves --workspace=.. --clean_dir=path_to_clean_wav --noisy_dir=path_to_noisy_wav

Then run train.py to train the U-net, command line:

python train.py
