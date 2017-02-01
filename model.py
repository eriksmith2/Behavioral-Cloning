# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 18:32:22 2017

@author: Erik Smith
"""
#package imports
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

#take image file paths and build a numpy array of each input image's pixel data
def read_in_images(files):
    images = np.zeros((len(files), 160, 320, 3), dtype = np.float32)
    for i, img in enumerate(files):
        images[i] = io.imread(img)
    return images

#image generator. The full image set is too large to fit into memory.
#takes in a list of image paths, their corresponding labels and a batch size
#outputs an array of images and an array of the steering angle captured with the image.
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
#function to resize images to the dimensions expected by the Nvidia architecture
def resize(image):
    import tensorflow as tf
    return tf.image.resize_images(image, (66, 200))

#normalization of the image data to between -0.5 and 0.5; yields better results in training.
def normalize(image):
    return image / 255.0 - 0.5

# Keras Imports
from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D

# Create the Sequential model. Model architecture is based on
# Nvidia's "End to End Learning for Self-Driving Cars" architecture as outlined
# in http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
# A cropping layer was added to trim some of the unnecessary sky portion of images.

model = Sequential()

model.add(Cropping2D(cropping=((22, 1), (1, 1)), input_shape=(160, 320, 3)))
model.add(Lambda(resize))
model.add(Lambda(normalize))
#The above three layers were added to preprocess the images. Doing the preprocessing
#inside the model ensures that feed data from drive.py is processed in the same way
#as the training data.
#model.add(BatchNormalization())
#removed since it seemed to be having limited impact during testing.
model.add(Convolution2D(24, 5, 5, activation='relu', border_mode='valid', subsample=(2,2)))
model.add(Convolution2D(36, 5, 5, activation='relu', border_mode='valid', subsample=(2,2)))
model.add(Convolution2D(48, 5, 5, activation='relu', border_mode='valid', subsample=(2,2)))
model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='valid', subsample=(1,1)))
model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='valid', subsample=(1,1)))
#model.add(Dropout(0.5))
#removed dropout, was causing negative results during testing.
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
#model.add(Dropout(0.5))
#removed dropout, was causing negative results during testing.
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

#output model summary for troubleshooting
model.summary()

# Training the network
#compile using Adam optimizer and mean-square-error goal
model.compile('Adam', 'mse')

#data generator inputs
data = get_data(image_paths, labels, batch_size = 64 )

history = model.fit_generator(data, samples_per_epoch = len(image_paths), nb_epoch = 5)

#save out trained model and weights.
json_string = model.to_json()
with open('model.json', 'w') as f:
    json.dump(json_string, f)
model.save_weights('model.h5')
print('model saved')
