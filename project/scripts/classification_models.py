#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 20:08:31 2022

@author: ezamoraa
"""

from matplotlib.pyplot import *
from sklearn import model_selection, tree
import sklearn.linear_model as lm
from toolbox_02450 import mcnemar
import numpy as np

from data import *


def get_classifier_error(y_exp, y_est):
    return sum(y_est != y_exp) / float(len(y_est))


def train_eval_logistic_regression(D_train, D_test, reg_strength=1.0):
    x_train, y_train = D_train
    x_test, y_test = D_test

    model = lm.LogisticRegression(C=1./reg_strength)
    model = model.fit(x_train, y_train)

    y_est_test = model.predict(x_test)
    y_est_train = model.predict(x_train)
    # y_est_prob_0 = model.predict_proba(x_test)[:, 0]

    error_test = get_classifier_error(y_test, y_est_test)
    error_train = get_classifier_error(y_train, y_est_train)
    return model, (error_test, error_train)


def train_eval_decision_tree(D_train, D_test, criterion="gini", max_depth=None):
    x_train, y_train = D_train
    x_test, y_test = D_test
    # Fit decision tree classifier, different pruning levels
    model = tree.DecisionTreeClassifier(criterion=criterion, max_depth=max_depth)
    model = model.fit(x_train, y_train)

    # Evaluate classifier's misclassification rate over train/test data
    y_est_test = np.asarray(model.predict(x_test), dtype=int)
    y_est_train = np.asarray(model.predict(x_train), dtype=int)
    error_test = get_classifier_error(y_test, y_est_test)
    error_train = get_classifier_error(y_train, y_est_train)
    return model, (error_test, error_train)


def train_eval_baseline(D_train, D_test):
    x_train, y_train = D_train
    x_test, y_test = D_test
    classes = list(classDict.values())
    cid = np.argmax([sum(y == c) for c in classes])

    class BaselineClassifier:
        def __init__(self, y_est):
            self.y_est = y_est

        def predict(self, x):
            return np.array([self.y_est] * len(x))

    model = BaselineClassifier(classes[cid])

    y_est_test = model.predict(x_test)
    y_est_train = model.predict(x_train)
    error_test = get_classifier_error(y_test, y_est_test)
    error_train = get_classifier_error(y_train, y_est_train)

    return model, (error_test, error_train)


def decision_tree_gvz(model, k, t):
    name = "gvz/tree_k{}_tc{}.gvz".format(k1, t)
    out = tree.export_graphviz(model, out_file=name, feature_names=attributeNames)
    import graphviz

    return graphviz.Source.from_file(name)


if __name__ == "__main__":
    ## Two-level crossvalidation
    # Create crossvalidation partition for evaluation
    K1 = K2 = 10
    CV1 = model_selection.KFold(n_splits=K1, shuffle=True)
    CV2 = model_selection.KFold(n_splits=K2, shuffle=True)

    # Number of model types (logistic regression, decision tree, baseline)
    nm = 3
    # Number of complexity parameters
    nc = 10

    # Best complexity parameter ID for each model
    S = np.zeros((K1, nm)).astype(int)
    # Estimated generalization error (outer cross-validation loop)
    Egen = np.zeros((K1, nm))

    # Logistic regression
    lr_m = 0  # Model ID for logistic regression (0 to nm-1)
    lr_c = [
        0.1 * (i + 1) for i in range(nc)
    ]  # Complexity parameter - regularization strength

    # Decision tree
    dt_c = [
        i + 1 for i in range(nc)
    ]  # Complexity parameter - constraint on maximum depth
    dt_criterion = "gini"
    dt_m = 1  # Model ID for decision tree (0 to nm-1)
    dt_gvz = []

    # Baseline
    bl_m = 2  # Model ID for baseline (0 to nm-1)

    # Center the data (subtract mean column values) and divide by the
    # attribute standard deviation to obtain a standardized dataset
    Xc = X - np.ones((N, 1)) * X.mean(0)
    Xc = Xc * (1 / np.std(Xc, 0))

    k1 = 0
    # Outer cross-validation loop
    for train_index1, test_index1 in CV1.split(X):
        # Extract training and test set for current CV fold
        x_train1 = Xc[train_index1, :]
        y_train1 = y[train_index1]
        x_test1 = Xc[test_index1, :]
        y_test1 = y[test_index1]

        D_train1 = (x_train1, y_train1)
        D_test1 = (x_test1, y_test1)

        Error_train = np.empty((K2, nm, nc))
        Error_test = np.empty((K2, nm, nc))
        length_test = np.empty((K2, 1))

        k2 = 0
        # Inner cross-validation loop
        for train_index2, test_index2 in CV2.split(x_train1):
            x_train2 = x_train1[train_index2, :]
            y_train2 = y_train1[train_index2]
            x_test2 = x_train1[test_index2]
            y_test2 = y_train1[test_index2]
            length_test[k2] = len(x_test2)

            D_train2 = (x_train2, y_train2)
            D_test2 = (x_test2, y_test2)

            # Evaluate multiple models with different complexity
            for c in range(nc):
                # Logistic regression
                model, err = train_eval_logistic_regression(D_train2, D_test2, lr_c[c])
                Error_test[k2][lr_m][c], Error_train[k2][lr_m][c] = err

                # Decision tree (method 2)
                model, err = train_eval_decision_tree(
                    D_train2, D_test2, dt_criterion, dt_c[c]
                )
                Error_test[k2][dt_m][c], Error_train[k2][dt_m][c] = err

                # Baseline
                model, err = train_eval_baseline(D_train2, D_test2)
                Error_test[k2][bl_m][c], Error_train[k2][bl_m][c] = err

            # print('CV2: Cross validation fold {0}/{1}'.format(k2+1,K2))
            # print('CV2: Train indices: {0}'.format(train_index2))
            # print('CV2: Test indices: {0}'.format(test_index2))
            k2 += 1

        Egen_c = np.empty((nm, nc))
        for m in range(nm):
            for c in range(nc):
                # Average of errors for (m=model_type, c=complexity) across inner cross validation sets
                Egen_c[m][c] = sum(
                    (length_test[k] / float(len(x_train1))) * Error_test[k][m][c]
                    for k in range(K2)
                )
            # Select best model complexity for each model type in this outer cross validation set
            S[k1][m] = np.argmin(Egen_c[m])

        # Train the optimal models again

        # Logistic regression
        s = S[k1][lr_m]
        model, err = train_eval_logistic_regression(D_train1, D_test1, lr_c[s])
        Egen[k1][lr_m], _ = err

        # Decision tree (method 2)
        s = S[k1][dt_m]
        model, err = train_eval_decision_tree(D_train1, D_test1, dt_criterion, dt_c[s])
        Egen[k1][dt_m], _ = err
        dt_gvz.append(decision_tree_gvz(model, k1, dt_c[s]))

        # Baseline
        s = S[k1][bl_m]  # Doesn't really matter (no complexity parameter)
        model, err = train_eval_baseline(D_train1, D_test1)
        Egen[k1][bl_m], _ = err

        print("CV1: Cross validation fold {0}/{1}".format(k1 + 1, K1))
        # print('CV1: Train indices: {0}'.format(train_index1))
        # print('CV1: Test indices: {0}'.format(test_index1))
        k1 += 1

    lr_c_opt = np.array([lr_c[S[k, lr_m]] for k in range(K1)])
    dt_c_opt = np.array([dt_c[S[k, dt_m]] for k in range(K1)])

    print("\nGeneralization error for outer folds (K1):")
    print(
        "  - Logistic regression (avg={}):\n    {}".format(
            np.mean(Egen[:, lr_m]), Egen[:, lr_m]
        )
    )
    print(
        "  - Decision tree (avg={}):\n    {}".format(
            np.mean(Egen[:, dt_m]), Egen[:, dt_m]
        )
    )
    print(
        "  - Baseline (avg={}):\n    {}".format(np.mean(Egen[:, bl_m]), Egen[:, bl_m])
    )

    print("\nOptimal model complexity parameter for outer folds (K1):")
    print("  - Logistic regression:\n    ", lr_c_opt)
    print("  - Decision tree:\n    ", dt_c_opt)
