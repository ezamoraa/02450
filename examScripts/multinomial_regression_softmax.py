#!/usr/bin/env python3

# Q24, Fall 2019

# First class weights
w1 = np.array([0.04, 1.32, -1.48])
# Second class weights
w2 = np.array([-0.03, 0.7, -0.85])
# Note: Third class is calculated differently

X = np.array([1, -5.52, -4.69])

f1 = np.array([X @ w1.T, X @ w2.T])
ef1 = np.exp(f1)

# It generates the output probability vector for 3 classes
y = (1/(1 + np.sum(ef1)))*np.hstack((ef1, np.array([1])))
