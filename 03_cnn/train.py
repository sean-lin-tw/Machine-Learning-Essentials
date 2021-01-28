#!/usr/bin/env python
# coding: utf-8

import os
import shutil
import math
import numpy as np
from keras.models import Sequential  # NN Activation
from keras.layers import Conv2D  # Convolution Operation
from keras.layers import MaxPooling2D # Pooling
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Dense # Fully Connected Networks
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from argparse import ArgumentParser


'''
Reading input data (training, validation, test)
'''
parser = ArgumentParser()
parser.add_argument('dir_data', help='Directory path that contains training, validation, and testing data')
args = parser.parse_args()
dir_train = os.path.join(args.dir_data, 'training')
dir_validation = os.path.join(args.dir_data, 'validation')

# Grouping training data for loading direclt from directories
for filename in os.listdir(dir_train):
    if filename.endswith(".jpg"):
        group = os.path.splitext(filename)[0].split('_')[0]
        dir_group = os.path.join(dir_train, group)
        if not os.path.isdir(dir_group):
            os.mkdir(dir_group) 
        # Move a file by renaming it's path
        shutil.move(os.path.join(dir_train, filename), os.path.join(dir_group, filename))

# Grouping validation data for loading direclt from directories
for filename in os.listdir(dir_validation):
    if filename.endswith(".jpg"):
        group = os.path.splitext(filename)[0].split('_')[0]
        dir_group = os.path.join(dir_validation, group)
        if not os.path.isdir(dir_group):
            os.mkdir(dir_group) 

        shutil.move(os.path.join(dir_validation, filename), os.path.join(dir_group, filename))

# Loading training/validation data from directories and fitting images to the CNN
batch_size = 32    
img_size = (224, 224)
train_datagen = ImageDataGenerator(
                                    rescale = 1./255,
                                    rotation_range=40,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2, 
                                    shear_range = 0.2, 
                                    zoom_range = 0.2, 
                                    horizontal_flip = True)                                   
validation_datagen = ImageDataGenerator(rescale = 1./255)
training_data = train_datagen.flow_from_directory(dir_train, target_size = img_size, batch_size = batch_size)
validation_data = validation_datagen.flow_from_directory(dir_validation, target_size = img_size, batch_size = batch_size)


'''
Building a CNN:
    Convolution layer -> Max pooling -> Convolution layer -> Max pooling -> ... -> Convolution layer -> Max pooling -> Flattening
'''

'''
Custom model

# Initializing CNN
model = Sequential()  

# First convolutional layer (and max pooling)
model.add(Conv2D(32, (3, 3), input_shape = (img_size[0], img_size[1], 3), activation = 'relu'))
model.add(Conv2D(32, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

# Second convolutional layer (and max pooling)
model.add(Conv2D(64, (3, 3), activation = 'relu'))
model.add(Conv2D(64, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(128, (3, 3), activation = 'relu'))
model.add(Conv2D(128, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
# Dropout layer prevents overfitting
# model.add(Dropout(0.25))

# Flattening
model.add(Flatten())

# Feedforward Neural Network
model.add(Dense(units = 128, activation = 'relu'))
model.add(Dense(units = 11, activation= 'softmax'))

# Compile CNN
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
'''

'''
VGG-16
'''
model = Sequential()  

model.add(Conv2D(64, (3, 3), input_shape = (img_size[0], img_size[1], 3), padding = "same", activation = 'relu'))
model.add(Conv2D(64, (3, 3), padding = "same", activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2), strides=(2,2)))

model.add(Conv2D(128, (3, 3), padding = "same", activation = 'relu'))
model.add(Conv2D(128, (3, 3), padding = "same", activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2), strides=(2,2)))

model.add(Conv2D(256, (3, 3), padding = "same", activation = 'relu'))
model.add(Conv2D(256, (3, 3), padding = "same", activation = 'relu'))
model.add(Conv2D(256, (3, 3), padding = "same", activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2), strides=(2,2)))

model.add(Conv2D(512, (3, 3), padding = "same", activation = 'relu'))
model.add(Conv2D(512, (3, 3), padding = "same", activation = 'relu'))
model.add(Conv2D(512, (3, 3), padding = "same", activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2), strides=(2,2)))

model.add(Conv2D(512, (3, 3), padding = "same", activation = 'relu'))
model.add(Conv2D(512, (3, 3), padding = "same", activation = 'relu'))
model.add(Conv2D(512, (3, 3), padding = "same", activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2), strides=(2,2)))

model.add(Flatten())

model.add(Dense(units = 4096, activation="relu"))
model.add(Dense(units = 4096, activation="relu"))
model.add(Dense(units = 11, activation= 'softmax'))

opt = Adam(lr=0.001)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# Show summary of the CNN network
model.summary()


'''
Training CNN
'''
steps_per_epoch = math.ceil(9866 / batch_size)
validation_steps = math.ceil(3430 / batch_size)
epochs = 100
history = model.fit_generator(training_data, steps_per_epoch = steps_per_epoch, epochs = epochs, validation_data = validation_data, validation_steps = validation_steps)

# Evaluate model
print("Evaluate on training data")
results_training = model.evaluate_generator(train_datagen)
print("test loss, test acc:", results_training)
print("Evaluate on training data")
results_validation = model.evaluate_generator(validation_datagen)
print("test loss, test acc:", results_validation)

# Saving model
model.save('hw3.h5')
