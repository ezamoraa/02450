# exercise 8.2.6
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
y = X[:, lm_idx].reshape(-1,1)
X1 = X[:,:lm_idx]
X2 = X[:,lm_idx+1:]
X = np.concatenate((X1, X2), axis=1)
# X = X[:, 0].reshape(-1,1)

attributeNames.pop(lm_idx)
N, M = X.shape

# Normalize data
# X = stats.zscore(X)
# y = stats.zscore(y)
                

def get_classifier_error(y_exp, y_est):
    return sum((y_est - y_exp)**2) / float(len(y_est)) 


def train_eval_neural_network(D_train, D_test, hidden_units=1):   
    x_train, y_train = D_train
    x_test, y_test = D_test
    
    x_train = torch.Tensor(x_train)
    y_train = torch.Tensor(y_train)
    x_test = torch.Tensor(x_test)
    y_test = torch.Tensor(y_test)
    
    n_hidden_units = hidden_units        # number of hidden units
    n_replicates = 2        # number of networks trained in each k-fold
    max_iter = 100000

    model = lambda: torch.nn.Sequential(
                        torch.nn.Linear(M, n_hidden_units), #M features to n_hidden_units
                        # torch.nn.Tanh(),   # 1st transfer function,
                        torch.nn.ReLU(),    # 1st transfer function, ReLU, Tanh, 
                        torch.nn.Linear(n_hidden_units, 1), # n_hidden_units to 1 output neuron
                        # no final tranfer function, i.e. "linear output"
                        )
    loss_fn = torch.nn.MSELoss() # notice how this is now a mean-squared-error loss

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
    
    error_test = get_classifier_error(y_test, y_est_test)
    error_train = get_classifier_error(y_train, y_est_train)
    
    return model, (error_test, error_train)



# def train_eval_baseline(D_train, D_test):
#     x_train, y_train = D_train
#     x_test, y_test = D_test

#     classes = list(classDict.values())
#     cid = np.mean(y_test) for c in classes

#     class BaselineClassifier:
#         def __init__(self, y_est):
#             self.y_est = y_est

#         def predict(self, x):
#             return np.array([self.y_est] * len(x))

#     model = BaselineClassifier(classes[cid])

#     y_est_test = model.predict(x_test)
#     y_est_train = model.predict(x_train)
#     error_test = get_classifier_error(y_test, y_est_test)
#     error_train = get_classifier_error(y_train, y_est_train)
    
#     return model, (error_test, error_train)



def main():
    
    K1 = 2
    CV1 = model_selection.KFold(n_splits=K1, shuffle=True)
    
    nc = 10
    
    hidden_units = 5
    
    for train_index1, test_index1 in CV1.split(X):
        
        x_train1 = X[train_index1, :]
        y_train1 = y[train_index1]
        x_test1 = X[test_index1, :]
        y_test1 = y[test_index1]
        
        D_train1 = (x_train1, y_train1)
        D_test1 = (x_test1, y_test1)
        
        model, error = train_eval_neural_network(D_train1, D_test1, hidden_units)
        

if __name__ == "__main__":
    main()







def correlated_ttest(r, rho, alpha=0.05):
    rhat = np.mean(r)
    shat = np.std(r)
    J = len(r)
    sigmatilde = shat * np.sqrt(1 / J + rho / (1 - rho))

    CI = st.t.interval(1 - alpha, df=J - 1, loc=rhat, scale=sigmatilde)  # Confidence interval
    p = 2*st.t.cdf(-np.abs(rhat) / sigmatilde, df=J - 1)  # p-value
    return p, CI



# statistical analysis; 1 is base line vs LR, 2 is base line vs ANN, 3 is LR vs ANN
alpha = 0.05
rho = 1/K
p,CI = np.zeros(3),np.zeros((3,2))
for i in range(3):
    p[i], CI[i,:] = correlated_ttest(r[:,i],rho,alpha=alpha)












