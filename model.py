# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 18:32:22 2017

@author: Erik
"""
import numpy as np
import pandas as pd
from skimage import io
import json

# Data Import

log_path = 'C:/Users/Erik/Desktop/Default Windows desktop 64-bit_Data/Training Data/driving_log.csv'
image_path = 'C:/Users/Erik/Desktop/Default Windows desktop 64-bit_Data/Training Data/'
raw_data = pd.read_csv(log_path, header = 0)
raw_data = raw_data.as_matrix(['center','steering'])
np.random.shuffle(raw_data)
image_names = raw_data[:,0]
image_paths =  image_path + image_names
labels = raw_data[:,1]

def read_in_images(files):
    images = np.zeros((len(files), 160, 320, 3), dtype = np.float32)
    for i, img in enumerate(files):
        images[i] = io.imread(img)
    return images

def get_data(image_paths, labels, batch_size):
    X = np.zeros((batch_size, 160, 320, 3), dtype = np.float32)
    y = np.zeros((batch_size,), dtype = np.float32)
    while 1:
        i = 0
        while i < len(image_paths):
            X = read_in_images(image_paths[i:i+batch_size])
            y = labels[i:i+batch_size]
            yield X,y

            if i + batch_size > len(image_paths):
                i = 0

            else:
                i += batch_size

def resize(image):
    import tensorflow as tf  # This import is required here otherwise the model cannot be loaded in drive.py
    return tf.image.resize_images(image, (66, 200))

def normalize(image):
    return image / 255.0 - 0.5


# Keras Imports
from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Lambda, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.callbacks import TensorBoard, EarlyStopping

# Create the Sequential model. Model architecture is based on the
# Nvidia's "End to End Learning for Self-Driving Cars" architecture as outlined
# in http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf

model = Sequential()

model.add(Cropping2D(cropping=((22, 1), (1, 1)), input_shape=(160, 320, 3)))
model.add(Lambda(resize, input_shape=(160, 320, 3)))
model.add(Lambda(normalize))
#model.add(BatchNormalization())
model.add(Convolution2D(24, 5, 5, activation='relu', border_mode='valid', subsample=(2,2)))
model.add(Convolution2D(36, 5, 5, activation='relu', border_mode='valid', subsample=(2,2)))
model.add(Convolution2D(48, 5, 5, activation='relu', border_mode='valid', subsample=(2,2)))
model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='valid', subsample=(1,1)))
model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='valid', subsample=(1,1)))
#model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

model.summary()

# Training the network
model.compile('Adam', 'mse')

data = get_data(image_paths, labels, batch_size = 64 )

#tensor_board = TensorBoard(log_dir='C:/Users/Erik/Dropbox/School/Self Driving Car/Behavioral Cloning/logs', histogram_freq=1, write_graph=True, write_images=True)

#stop = EarlyStopping(monitor='loss', min_delta=0.001, patience=1, verbose=0, mode='auto')

history = model.fit_generator(data,
                              samples_per_epoch = len(image_paths),
                              nb_epoch = 5#,
                              #callbacks = [stop]
                              )

json_string = model.to_json()
with open('model.json', 'w') as f:
    json.dump(json_string, f)
model.save_weights('model.h5')
print('model saved')
