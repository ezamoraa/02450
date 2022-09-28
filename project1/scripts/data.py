#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 09:35:57 2022

@author: ezamoraa
"""

import numpy as np
import xlrd

doc = xlrd.open_workbook('../dataset/Rice_Cammeo_Osmancik.xls').sheet_by_index(0)

# Extract attribute names
attributeNames = doc.row_values(rowx=0, start_colx=0, end_colx=7)

num_features = 7
num_entries = 3810
class_col = num_features

classLabels = doc.col_values(class_col,1,num_entries+1)
classNames = sorted(set(classLabels))
classDict = dict(zip(classNames,range(len(classNames))))

# Extract vector y, convert to NumPy array
y = np.array([classDict[value] for value in classLabels])

# Preallocate memory, then extract data to matrix X
X = np.empty((num_entries,num_features))
for i in range(num_features):
    X[:,i] = np.array(doc.col_values(i,1,num_entries+1)).T

# Compute values of N, M and C.
N = len(y)
M = len(attributeNames)
C = len(classNames)

# Random shuffle data
# p = np.random.permutation(N)
# X, y = X[p], y[p]
