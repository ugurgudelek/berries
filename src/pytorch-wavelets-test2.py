# -*- coding: utf-8 -*-
# @Time   : 3/28/2020 6:20 AM
# @Author : Ugur Gudelek
# @Email  : ugurgudelek@gmail.com
# @File   : pytorch-wavelets-test2.py

# MIT License
#
# Copyright (c) 2018 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Tom Runia
# Date Created: 2018-04-16

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from wavelets_pytorch.transform import WaveletTransform
from wavelets_pytorch.transform import WaveletTransformTorch

from plot_scalogram import plot_scalogram

######################################

fps = 8000
dt  = 1.0/fps
dj  = 0.125
unbias = False


batch_size = 128
duration = 1  # sec
signal_lengths = duration*fps

current_time = time.time()

t = np.linspace(0, duration, duration*fps)
random_frequencies = np.random.uniform(-0.5, 4.0, size=batch_size)
batch = np.asarray([np.sin(2*np.pi*f*t) for f in random_frequencies])

print(batch.shape)

# Perform batch computation of Torch implementation
wa = WaveletTransformTorch(dt, dj, unbias=unbias)


power = wa.power(batch)
print(f"Elapsed time: {time.time() - current_time}")
current_time = time.time()


fig, ax = plt.subplots(1, 2, figsize=(12, 3))
ax = ax.flatten()
ax[0].plot(t, batch[0])
ax[0].set_title(r'$f(t) = \sin(2\pi \cdot f t) + \mathcal{N}(\mu,\,\sigma^{2})$')
ax[0].set_xlabel('Time (s)')

# Plot scalogram for PyTorch implementation
plot_scalogram(power[0], wa.fourier_periods, t, ax=ax[1])
ax[1].axhline(1.0 / random_frequencies[0], lw=1, color='k')
ax[1].set_title('Scalogram (Torch)'.format(1.0/random_frequencies[0]))
ax[1].set_ylabel('')
ax[1].set_yticks([])

plt.tight_layout()
print(f"Elapsed time: {time.time() - current_time}")
plt.show()

