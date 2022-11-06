#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 13:30:06 2022

@author: ezamoraa
"""

from matplotlib.pyplot import *
from sklearn import model_selection, tree
import sklearn.linear_model as lm
from toolbox_02450 import mcnemar
import numpy as np

from data import *
from classification_models import *

# Logistic regression complexity parameter
C = 0.2
# Center the data (subtract mean column values) and divide by the
# attribute standard deviation to obtain a standardized dataset
Xc = X - np.ones((N, 1)) * X.mean(0)
Xc = Xc * (1 / np.std(Xc, 0))

D_train = D_test = (Xc, y)
model, err = train_eval_logistic_regression(D_train, D_test, C)
