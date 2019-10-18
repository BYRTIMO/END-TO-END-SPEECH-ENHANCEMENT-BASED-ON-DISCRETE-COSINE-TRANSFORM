# -*- coding: utf-8 -*-

workdir = '..'

#
# Network Optimization
#
lr = 1e-4       # Initial learning rate
min_lr = 1e-5     # Learning rate minimum

#
# Define short-term spectral parameters
#
sample_rate = 16000         # Sampling Rate
n_window = 1024             # Window length
stride = 64              # Step size
selection = 9152            # sequence length

#
# Network training learning rate adjustment 
# strategy and early stop strategy
#
patience = 40  # Endurance limit
cooldown = 20   # The number of epochs that keep the learning rate constant
factor = 0.4   # Learning rate decay factor
start_it = 10  # The number of iterations at which the learning rate begins to adjust 
cnt = 20        # Verify that the set loss value stops falling

#
# 
#
batch_size = 16