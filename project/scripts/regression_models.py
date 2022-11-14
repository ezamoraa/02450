from matplotlib.pylab import *
import numpy as np
from scipy.io import loadmat
import torch
from sklearn import model_selection
from toolbox_02450 import train_neural_net, draw_neural_net
from scipy import stats
from data import *


from sklearn.model_selection import train_test_split
#X_train, X, y_train, giy = train_test_split(X, y, test_size=.005, stratify=y)
    
# Preparing data  
lm_idx = attributeNames.index('Area')
y = X[:, lm_idx].reshape(-1, 1)
X1 = X[:, :lm_idx]
X2 = X[:, lm_idx+1:]
X = np.concatenate((X1, X2), axis=1)
# X = X[:, 0].reshape(-1,1)
attributeNames.pop(lm_idx)

# Removing convex area when predicting area
rm_idx = attributeNames.index('Convex_Area')
attributeNames.pop(rm_idx)
Xrm1 = X[:, :rm_idx]
Xrm2 = X[:, rm_idx+1:]
X = np.concatenate((Xrm1, Xrm2), axis=1)


N, M = X.shape


def get_regression_error(y_exp, y_est):
    return np.sum(np.power(y_exp - y_est.reshape(-1,1), 2)) / float(len(y_est))

def get_squared_error(y_exp, y_est):
    #z = np.power(np.abs(y_exp - y_est.reshape(-1,1)),2) #Squared error
    z = np.abs(y_exp - y_est.reshape(-1,1)) #Absolute error
    return z

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
    
    z = get_squared_error(y_test, y_est_test)

    return model, z, (error_test, error_train)


def train_eval_neural_network(D_train, D_test, hidden_units=1):
    x_train, y_train = D_train
    x_test, y_test = D_test
    
    # Normalize data
    # x_train = stats.zscore(x_train)
    # x_test = stats.zscore(x_test)
    # y = stats.zscore(y)
    

    x_train = torch.Tensor(x_train)
    y_train = torch.Tensor(y_train)
    x_test = torch.Tensor(x_test)
    y_test = torch.Tensor(y_test)

    n_hidden_units = hidden_units        # number of hidden units
    n_replicates = 2        # number of networks trained in each k-fold
    max_iter = 30000

    def model(): return torch.nn.Sequential(
        torch.nn.Linear(M, n_hidden_units),  # M features to n_hidden_units
        #torch.nn.Tanh(),   # 1st transfer function,
        ##torch.nn.ReLU(),    # 1st transfer function, ReLU, Tanh,
        #torch.nn.Linear(n_hidden_units, n_hidden_units),  # n_hidden_units to n_hidden_units
        torch.nn.ReLU(),    # 2nd transfer function
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
    y_est_train = net(x_train).data.numpy()
    y_est_test = net(x_test).data.numpy()

    y_train = y_train.data.numpy()
    y_test = y_test.data.numpy()

    error_train = get_regression_error(y_train, y_est_train)
    error_test = get_regression_error(y_test, y_est_test)
    
    z = get_squared_error(y_test, y_est_test)

    return model, z, (error_test, error_train)


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
    
    z = get_squared_error(y_test, y_est_test)

    return model, z, (error_test, error_train)


def stat_eval_ttest(z_all, nm):
    z_rlr = z_all[:,0]
    z_ann = z_all[:,1]
    z_bl = z_all[:,2]
    
    alpha = 0.05
    
    z = []    
    z.append(z_ann - z_rlr)
    z.append(z_ann - z_bl)
    z.append(z_rlr - z_bl)
    
    CI = []
    p = []
    
    for i in range(nm):
        CI.append( stats.t.interval(1-alpha, len(z[i])-1, loc=np.mean(z[i]), scale=stats.sem(z[i])) ) # Confidence interval
        p.append( 2*stats.t.cdf( -np.abs( np.mean(z[i]) )/stats.sem(z[i]), df=len(z[i])-1) )  # p-value
    
    return p, CI

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
        np.power(10., i-5) for i in range(nc)
    ])  # Complexity parameter - regularization strength
    rlr_m = 0

    # ANN complexity parameter
    ann_c = np.array([
        i + 1 for i in range(nc)
    ])  # Complexity parameter - regularization strength
    ann_m = 1

    # Baseline
    bl_m = 2  # Model ID for baseline (0 to nm-1)

    k1 = 0    
    
    z_all = []
    # Outer cross-validation loop
    for train_index1, test_index1 in CV1.split(X):
        print("CV1: Cross validation fold {0}/{1}".format(k1 + 1, K1))
        
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
                model, _ , err = train_eval_regularized_reg(D_train2, D_test2, rlr_c[c], attributeNames, M)
                Error_test[k2][rlr_m][c], Error_train[k2][rlr_m][c] = err

                # ANN
                model, _ , err = train_eval_neural_network(D_train2, D_test2, ann_c[c])
                Error_test[k2][ann_m][c], Error_train[k2][ann_m][c] = err

                # Baseline
                model, _ , err = train_eval_baseline(D_train2, D_test2)
                Error_test[k2][bl_m][c], Error_train[k2][bl_m][c] = err
                
            figure(k2*(k1+1), figsize=(12,8))
            plot(ann_c, Error_test[k2][ann_m])
            #title('Evaluation of number of hidden units for ANN')
            grid()
            xlabel('Amount of hidden units')
            ylabel('Error (MSE)')

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


        z=[]
        # Train the optimal models again
        # RLR
        s = S[k1][rlr_m]
        model, z_m, err = train_eval_regularized_reg(D_train1, D_test1, rlr_c[s], attributeNames, M)
        Egen[k1][rlr_m], _ = err
        z.append(z_m[:,0]) #Add errors for statistical evaluation

        # ANN
        s = S[k1][ann_m]
        model, z_m, err = train_eval_neural_network(D_train1, D_test1, ann_c[s])
        Egen[k1][ann_m], _ = err
        z.append(z_m[:,0]) #Add errors for statistical evaluation

        # Baseline
        s = S[k1][bl_m]
        model, z_m, err = train_eval_baseline(D_train1, D_test1)
        Egen[k1][bl_m], _ = err
        z.append(z_m[:,0]) #Add errors for statistical evaluation
        
        #To compare errors (for statistical evaluation)
        z = np.stack(z, axis=1)
        z_all.append(z)

        #print("CV1: Cross validation fold {0}/{1}".format(k1 + 1, K1))
        # print('CV1: Train indices: {0}'.format(train_index1))
        # print('CV1: Test indices: {0}'.format(test_index1))
        k1 += 1

    z_all = np.concatenate(z_all) #Concatenating to compare errors for statistical evaluation
    p, CI = stat_eval_ttest(z_all, nm)
    

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
    
    
    print(" ----- Statistical evaluation ------ ")
    print("ANN vs RLR  ---  CI:", CI[0], "   ---   p: ", p[0])
    print("ANN vs BL   ---  CI:", CI[1], "   ---   p: ", p[1])
    print("RLR vs BL   ---  CI:", CI[2], "   ---   p: ", p[2])
    
    
    


if __name__ == "__main__":
    main()