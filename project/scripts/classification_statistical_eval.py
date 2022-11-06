#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 12:54:02 2022

@author: ezamoraa
"""

from matplotlib.pyplot import *
from sklearn import model_selection, tree
import sklearn.linear_model as lm
from toolbox_02450 import mcnemar
import numpy as np

from data import *
from classification_models import *


def main():
    # Number of model types (logistic regression, decision tree, baseline)
    nm = 3
    lr_m = 0  # Model ID for logistic regression (0 to nm-1)
    dt_m = 1  # Model ID for decision tree (0 to nm-1)
    bl_m = 2  # Model ID for baseline (0 to nm-1)

    ## Statistical evaluation (pairwise)
    K = 10
    # Perform one-level crossvalidation
    CV = model_selection.KFold(n_splits=K, shuffle=True)
    # CV = model_selection.LeaveOneOut()
    i = 0

    # Center the data (subtract mean column values) and divide by the
    # attribute standard deviation to obtain a standardized dataset
    Xc = X - np.ones((N, 1)) * X.mean(0)
    Xc = Xc * (1 / np.std(Xc, 0))

    yhat = []
    y_true = []
    for train_index, test_index in CV.split(X, y):
        print("Crossvalidation fold: {0}/{1}".format(i + 1, K))

        # extract training and test set for current CV fold
        X_train = Xc[train_index, :]
        y_train = y[train_index]
        X_test = Xc[test_index, :]
        y_test = y[test_index]

        D_train = (X_train, y_train)
        D_test = (X_test, y_test)

        dy = []

        train_eval_model_fns = [
            train_eval_logistic_regression,
            train_eval_decision_tree,
            train_eval_baseline,
        ]
        for train_eval_model_fn in train_eval_model_fns:
            model, _ = train_eval_model_fn(D_train, D_test)
            y_est = model.predict(X_test)
            dy.append(y_est)

        dy = np.stack(dy, axis=1)
        yhat.append(dy)
        y_true.append(y_test)

        i += 1

    yhat = np.concatenate(yhat)
    y_true = np.concatenate(y_true)

    acc = []  # Accuracy
    for m in range(nm):
        acc_m = np.sum(yhat[:, m] == y_true) / len(y_true)
        acc.append(acc_m)

    alpha = 0.01
    print("\nStatistical evaluation [logistic regression - decision tree]:")
    [thetahat, CI, p] = mcnemar(y_true, yhat[:, lr_m], yhat[:, dt_m], alpha=alpha)
    print(
        "  theta = theta_A-theta_B point estimate", thetahat, " CI: ", CI, "p-value", p
    )

    print("\nStatistical evaluation [logistic regression - baseline]:")
    [thetahat, CI, p] = mcnemar(y_true, yhat[:, lr_m], yhat[:, bl_m], alpha=alpha)
    print(
        "  theta = theta_A-theta_B point estimate", thetahat, " CI: ", CI, "p-value", p
    )

    print("\nStatistical evaluation [decision tree - baseline]:")
    [thetahat, CI, p] = mcnemar(y_true, yhat[:, dt_m], yhat[:, bl_m], alpha=alpha)
    print(
        "  theta = theta_A-theta_B point estimate", thetahat, " CI: ", CI, "p-value", p
    )


if __name__ == "__main__":
    main()
