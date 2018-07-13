#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 13:08:27 2018

@author: keyur-r
"""

# CNN classifier

# Building architecture of our CNN classifier
import keras
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Step - 1 Convolution
classifier.add(Convolution2D(
    16, 3, 3, input_shape=(28, 28, 3), activation='relu'))

# Step - 2 Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Convolution2D(32, 3, 3, activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Step - 3 Flattening
classifier.add(Flatten())

# Step - 4 Full connection -> First layer input layer then hidden layer
# and last softmax layer
classifier.add(Dense(56, activation='relu', kernel_initializer='uniform'))
classifier.add(Dense(3, activation='softmax', kernel_initializer='uniform'))

# Compiling the CNN
classifier.compile(
    optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# Image Preprocessing
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

training_set = train_datagen.flow_from_directory(
    'shapes/train', target_size=(28, 28), batch_size=1, class_mode='categorical')
#X_images, y_labels = training_set.filenames, training_set.classes
test_set = test_datagen.flow_from_directory(
    'shapes/test', target_size=(28, 28), batch_size=1, class_mode='categorical')

# Logging the training of models
from keras.callbacks import CSVLogger, EarlyStopping
csv_logger = CSVLogger('log.csv', append=True, separator=';')
early_stopping_monitor = EarlyStopping(patience=5)

steps_per_epoch = len(training_set.filenames)  # 300
validation_steps = len(test_set.filenames)  # 90

model_info = classifier.fit_generator(training_set, steps_per_epoch=steps_per_epoch, epochs=25, validation_data=test_set,
                                      validation_steps=validation_steps, callbacks=[csv_logger, early_stopping_monitor])

classifier.save("drawing_classification.h5")

# plot model history after each epoch
from visulization import plot_model_history
plot_model_history(model_info)
