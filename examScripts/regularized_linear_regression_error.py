#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 16:17:14 2022

@author: ezamoraa
"""

import numpy as np
from scipy.stats import zscore

X = np.array([2, 5, 6, 7]).reshape(-1,1)
Xs = zscore(X, ddof=1)

y = np.array([6, 7, 7, 9])
w = np.array([0.6])
w0 = np.mean(y)
l = 2

#E = 8
#l = (E - np.sum(np.square(y - w0 * np.ones(4) - Xs @ w))) / np.sum(w**2)

E = np.sum(np.square(y - w0 * np.ones(len(y)) - Xs @ w)) + l * np.sum(w**2)
