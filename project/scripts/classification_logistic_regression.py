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
from toolbox_02450 import rocplot, confmatplot
import numpy as np

from data import *
from classification_models import *

# Logistic regression regularization strength
lambda_ = 0.1
# Center the data (subtract mean column values) and divide by the
# attribute standard deviation to obtain a standardized dataset
Xc = X - np.ones((N, 1)) * X.mean(0)
Xc = Xc * (1 / np.std(Xc, 0))

y_train = y_test = y; X_train = X_test = Xc;
#X_train, X_test, y_train, y_test = model_selection.train_test_split(Xc,y,test_size=0.5)

D_train = (X_train, y_train); D_test = (X_test, y_test)

model, err = train_eval_logistic_regression(D_train, D_test, 1./lambda_)

y_est = model.predict(X_test).T
y_est_0_prob = model.predict_proba(X_test)[:,0]
y_est_1_prob = model.predict_proba(X_test)[:,1]

f = figure()
class0_ids = np.nonzero(y_test==0)[0].tolist()
plot(class0_ids, y_est_0_prob[class0_ids], '.y')
class1_ids = np.nonzero(y_test==1)[0].tolist()
plot(class1_ids, y_est_0_prob[class1_ids], '.r')
xlabel('Data object (rice grain sample)'); ylabel('Predicted prob. of class Cammeo');
legend(['Cammeo', 'Osmancik'])
ylim(-0.01,1.5)

show()


figure()
rocplot(y_est_1_prob,y_test)
show()

figure()
confmatplot(y_test,y_est)
show()

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_est)