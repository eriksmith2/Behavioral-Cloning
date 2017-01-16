# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 18:32:22 2017

@author: Erik
"""

#Imports
from keras.models import Sequential
from keras.layers.core import Dense, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution2D

# Create the Sequential model. Model architecture is based on the 
# Nvidia's "End to End Learning for Self-Driving Cars" architecture as outlined
# in http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf

model = Sequential()

# 1st Layer - Normalization layer
model.add(BatchNormalization(input_shape=(320, 160, 3)))

# 2nd Layer - 24 @ 31x98 Convolution
model.add(Convolution2D(24, 5, 5, border_mode='same'))

# 3rd Layer - 36 @ 16x47 Convolution
model.add(Convolution2D(36, 5, 5, border_mode='same'))

# 4th Layer - 48 @ 5x22 Convolution
model.add(Convolution2D(48, 5, 5, border_mode='same'))

# 5th Layer - 64 @ 3x20 Convolution
model.add(Convolution2D(64, 3, 3, border_mode='same'))

# 6th Layer - 64 @ 1x18 Convolution
model.add(Convolution2D(64, 3, 3, border_mode='same'))

# 7th Layer - Flatten
model.add(Flatten())

# 8th Layer - Fully Connected
model.add(Dense(100))

# 9th Layer - Fully Connected
model.add(Dense(50))

# 10th Layer - Fully Connected
model.add(Dense(1))