import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import torch
from sklearn import model_selection
from toolbox_02450 import train_neural_net, draw_neural_net
from scipy import stats
from data import *

# preparing data:
lm_idx = attributeNames.index('Area')
y = X[:, lm_idx].reshape(-1, 1)
X1 = X[:, :lm_idx]
X2 = X[:, lm_idx+1:]
X = np.concatenate((X1, X2), axis=1)
# X = X[:, 0].reshape(-1,1)

attributeNames.pop(lm_idx)
N, M = X.shape


def get_regression_error(y_exp, y_est):
    return np.sum(np.power(y_est - y_exp, 2)) / float(len(y_est))


def train_eval_regularized_reg(D_train, D_test, C, attributeNames, M):
    x_train, y_train = D_train
    x_test, y_test = D_test

    x_test = np.concatenate((np.ones((x_test.shape[0], 1)), x_test), 1)
    x_train = np.concatenate((np.ones((x_train.shape[0], 1)), x_train), 1)
    attributeNames = [u'Offset']+attributeNames

    # Standardize the training and set set based on training set moments
    mu = np.mean(x_train[:, 1:], 0)
    sigma = np.std(x_train[:, 1:], 0)

    x_train[:, 1:] = (x_train[:, 1:] - mu) / sigma
    x_test[:, 1:] = (x_test[:, 1:] - mu) / sigma

    # precompute terms
    Xty = x_train.T @ y_train
    XtX = x_train.T @ x_train

    lambdaI = C * np.eye(M+1)
    lambdaI[0, 0] = 0  # remove bias regularization
    # lambdaI[0,0] = 0 # remove bias regularization
    w = np.linalg.solve(XtX+lambdaI, Xty).squeeze()
    # Evaluate training and test performance½

    class RlrClassifier:
        def __init__(self, w):
            self.w = w

        def predict(self, x):
            return x @ self.w.T

    model = RlrClassifier(w)

    y_est_train = model.predict(x_train)
    y_est_test = model.predict(x_test)

    error_test = get_regression_error(y_test, y_est_test)
    error_train = get_regression_error(y_train, y_est_train)

    return model, (error_test, error_train)


def train_eval_neural_network(D_train, D_test, hidden_units=1):
    x_train, y_train = D_train
    x_test, y_test = D_test

    x_train = torch.Tensor(x_train)
    y_train = torch.Tensor(y_train)
    x_test = torch.Tensor(x_test)
    y_test = torch.Tensor(y_test)

    n_hidden_units = hidden_units        # number of hidden units
    n_replicates = 1        # number of networks trained in each k-fold
    max_iter = 10000

    def model(): return torch.nn.Sequential(
        torch.nn.Linear(M, n_hidden_units),  # M features to n_hidden_units
        # torch.nn.Tanh(),   # 1st transfer function,
        torch.nn.ReLU(),    # 1st transfer function, ReLU, Tanh,
        # n_hidden_units to 1 output neuron
        torch.nn.Linear(n_hidden_units, 1),
        # no final tranfer function, i.e. "linear output"
    )
    loss_fn = torch.nn.MSELoss()  # notice how this is now a mean-squared-error loss

    # Train the net on training data
    net, final_loss, learning_curve = train_neural_net(model,
                                                       loss_fn,
                                                       X=x_train,
                                                       y=y_train,
                                                       n_replicates=n_replicates,
                                                       max_iter=max_iter)
    y_est_test = net(x_test).data.numpy()
    y_est_train = net(x_train).data.numpy()

    y_test = y_test.data.numpy()
    y_train = y_train.data.numpy()

    error_test = get_regression_error(y_test, y_est_test)
    error_train = get_regression_error(y_train, y_est_train)

    return model, (error_test, error_train)


def train_eval_baseline(D_train, D_test):
    x_train, y_train = D_train
    x_test, y_test = D_test

    ytm = np.mean(y_test)

    class BaselineClassifier:
        def __init__(self, y_est):
            self.y_est = y_est

        def predict(self, x):
            return np.array([self.y_est] * len(x))

    model = BaselineClassifier(ytm)

    y_est_test = model.predict(x_test)
    y_est_train = model.predict(x_train)
    error_test = get_regression_error(y_test, y_est_test)
    error_train = get_regression_error(y_train, y_est_train)

    return model, (error_test, error_train)


def main():

    K1 = K2 = 10
    CV1 = model_selection.KFold(n_splits=K1, shuffle=True)
    CV2 = model_selection.KFold(n_splits=K2, shuffle=True)

    # Number of model types (RLR, ANN, baseline)
    nm = 3
    # Number of complexity parameters
    nc = 10

    # Best complexity parameter ID for each model
    S = np.zeros((K1, nm)).astype(int)
    # Estimated generalization error (outer cross-validation loop)
    Egen = np.zeros((K1, nm))

    # RLR complexity parameter
    rlr_c = np.array([
        np.power(10., i-2) for i in range(nc)
    ])  # Complexity parameter - regularization strength
    rlr_m = 0

    # ANN complexity parameter
    ann_c = np.array([
        i + 1 for i in range(nc)
    ])  # Complexity parameter - regularization strength
    ann_m = 0

    # Baseline
    bl_m = 2  # Model ID for baseline (0 to nm-1)

    k1 = 0
    # Outer cross-validation loop
    for train_index1, test_index1 in CV1.split(X):
        # Extract training and test set for current CV fold
        x_train1 = X[train_index1, :]
        y_train1 = y[train_index1]
        x_test1 = X[test_index1, :]
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
                # RLR
                model, err = train_eval_regularized_reg(D_train2, D_test2, rlr_c[c], attributeNames, M)
                Error_test[k2][rlr_m][c], Error_train[k2][rlr_m][c] = err

                # ANN
                model, err = train_eval_neural_network(D_train2, D_test2, ann_c[c])
                Error_test[k2][ann_m][c], Error_train[k2][ann_m][c] = err

                # Baseline
                model, err = train_eval_baseline(D_train2, D_test2)
                Error_test[k2][bl_m][c], Error_train[k2][bl_m][c] = err

            k2 += 1

        Egen_c = np.empty((nm, nc))
        for m in range(nm):
            for c in range(nc):
                # Average of errors for (m=model_type, c=complexity) across inner cross validation sets
                Egen_c[m][c] = sum(
                    (length_test[k] / float(len(x_train1))) *
                    Error_test[k][m][c]
                    for k in range(K2)
                )
            # Select best model complexity for each model type in this outer cross validation set
            S[k1][m] = np.argmin(Egen_c[m])

        # Train the optimal models again

        # RLR
        s = S[k1][rlr_m]
        model, err = train_eval_regularized_reg(D_train1, D_test1, rlr_c[s], attributeNames, M)
        Egen[k1][rlr_m], _ = err

        # ANN
        s = S[k1][ann_m]
        model, err = train_eval_neural_network(D_train1, D_test1, ann_c[s])
        Egen[k1][ann_m], _ = err

        # Baseline
        s = S[k1][bl_m]
        model, err = train_eval_baseline(D_train1, D_test1)
        Egen[k1][bl_m], _ = err


        print("CV1: Cross validation fold {0}/{1}".format(k1 + 1, K1))
        # print('CV1: Train indices: {0}'.format(train_index1))
        # print('CV1: Test indices: {0}'.format(test_index1))
        k1 += 1

    rlr_c_opt = np.array([rlr_c[S[k, rlr_m]] for k in range(K1)])
    ann_c_opt = np.array([ann_c[S[k, ann_m]] for k in range(K1)])

    print("\nGeneralization error for outer folds (K1):")
    print(
        "  - Regularized linear regression (avg={}):\n    {}".format(
            np.mean(Egen[:, rlr_m]), Egen[:, rlr_m]
        )
    )
    print(
        "  - ANN (avg={}):\n    {}".format(
            np.mean(Egen[:, ann_m]), Egen[:, ann_m]
        )
    )
    print(
        "  - Baseline (avg={}):\n    {}".format(
            np.mean(Egen[:, bl_m]), Egen[:, bl_m]
        )
    )

    print("\nOptimal model complexity parameter for outer folds (K1):")
    print("  - Regularized linear regression:\n    ", rlr_c_opt)
    print("  - ANN:\n    ", ann_c_opt)


if __name__ == "__main__":
    main()
