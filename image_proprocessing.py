#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 12:01:09 2018

@author: keyur-r
"""

# Image Preprocessing for train, test and validation sets

import os
import random
import glob


def prepare_test_data(n):
    base_path = "shapes"
    f1 = random.sample(glob.glob(os.path.join(base_path, "test/circles") + "/*"), n)
    f2 = random.sample(glob.glob(os.path.join(base_path, "test/squares") + "/*"), n)
    f3 = random.sample(glob.glob(os.path.join(base_path, "test/triangles") + "/*"), n)
    for c in f1:
        os.remove(c)
    for s in f2:
        os.remove(s)
    for t in f3:
        os.remove(t)

def prepare_validation_data(n):
    base_path = "shapes"
    f1 = random.sample(glob.glob(os.path.join(base_path, "validation/circles") + "/*"), n)
    f2 = random.sample(glob.glob(os.path.join(base_path, "validation/squares") + "/*"), n)
    f3 = random.sample(glob.glob(os.path.join(base_path, "validation/triangles") + "/*"), n)
    for c in f1:
        os.remove(c)
    for s in f2:
        os.remove(s)
    for t in f3:
        os.remove(t)

# n = number  of samples to remove from each categorical folder
prepare_test_data(70)
prepare_validation_data(40)