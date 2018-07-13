#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 14:01:56 2018

@author: keyur-r
"""

import pandas as pd
import numpy as np
import cv2
import argparse

from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, accuracy_score

from myutil import probas_to_classes


# Loading and compiling presaved trained CNN
model = load_model('drawing_classification.h5')

label = {0: "Circle", 1: "Square", 2: "Triangle"}


def predict_one(file_name):

    img = cv2.imread(file_name)
    img = cv2.resize(img, (28, 28))
    img = np.reshape(img, [1, 28, 28, 3])
    classes = model.predict_classes(img)[0]
    category = label[classes]
    print("\nAnd {1} is the {0}".format(category, file_name))
    # return category


def predict_dataset(input_dir):
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_generator = test_datagen.flow_from_directory("shapes/test",
                                                      target_size=(28, 28),
                                                      color_mode="rgb",
                                                      shuffle=False,
                                                      class_mode='categorical',
                                                      batch_size=1)
    filenames = test_generator.filenames
    nb_samples = len(filenames)
    predict = model.predict_generator(test_generator, steps=nb_samples)
    return predict, test_generator


def main():
    # Instantiate the parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--testdata', help='Classify images on test data', action='store_true')
    parser.add_argument(
        '--validationdata', help='Classify images on test data', action='store_true')

    parser.add_argument('--image', help='Input your image file name')
    args = parser.parse_args()

    on_dataset = False
    if args.testdata:
        print("Classify images on test dataset")
        on_dataset = True
        input_dir = "shapes/test"
    if args.validationdata:
        print("Classify images on validation dataset")
        on_dataset = True
        input_dir = "shapes/validation"

    if on_dataset:
        predict, test_generator = predict_dataset(input_dir)
        y_pred = probas_to_classes(predict)
        y_true = test_generator.classes
        X_images = test_generator.filenames
        cm = confusion_matrix(y_true, y_pred)
        ac = accuracy_score(y_true, y_pred)
        for ele in list(zip(X_images, y_true, y_pred)):
            print(ele)
    else:
        file_name = args.image
        predict_one(file_name)

if __name__ == '__main__':
    main()
