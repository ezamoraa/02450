import numpy as np
from sklearn import model_selection
from data import *
from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, 
                           title, subplot, show, grid)

area_idx = attributeNames.index('Area')
y_area = X[:, area_idx]

X_cols = list(range(0, area_idx)) + list(range(area_idx+1,len(attributeNames)))
X_without_area = X[:, X_cols]
N, M = X_without_area.shape


# Add offset attribute
X_without_area = np.concatenate((np.ones((X_without_area.shape[0],1)),X_without_area),1)
attributeNames = [u'Offset']+attributeNames
M = M+1

def train_eval_regularized_reg(D_train, D_test, C):

    x_train, y_train = D_train
    x_test, y_test = D_test

    # Standardize the training and set set based on training set moments
    mu = np.mean(x_train[:, 1:], 0)
    sigma = np.std(x_train[:, 1:], 0)
    
    x_train[:, 1:] = (x_train[:, 1:] - mu) / sigma
    x_test[:, 1:] = (x_test[:, 1:] - mu) / sigma
    
    # precompute terms
    Xty = x_train.T @ y_train
    XtX = x_train.T @ x_train
    
    lambdaI = C * np.eye(M)
    lambdaI[0,0] = 0 # remove bias regularization
    #lambdaI[0,0] = 0 # remove bias regularization
    w = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
    # Evaluate training and test performance

    def rlr_model(X_data):
        y_est = X_data @ w.T

        return y_est
    
    y_est_train = rlr_model(x_train)
    y_est_test = rlr_model(x_test)

    error_train = np.power(y_train-y_est_train,2).mean(axis=0)
    error_test = np.power(y_test-y_est_test,2).mean(axis=0)
    
    return w, error_test, error_train


nc = 10

# RLR complexity parameter
rlr_c = np.array([
    np.power(10., i-2) for i in range(nc)
])  # Complexity parameter - regularization strength
rlr_m = 0  # Update according to scripts with LMS


K = 5
CV = model_selection.KFold(n_splits=K, shuffle=True)
CV2 = model_selection.KFold(n_splits=K, shuffle=True)
lambdas = np.power(10.,range(-5,9))
w = np.empty((M,K,len(rlr_c)))
train_error = np.empty((K,len(rlr_c)))
test_error = np.empty((K,len(rlr_c)))

Error_train = np.empty((K,1))
Error_test = np.empty((K,1))
w_rlr = np.empty((M,K))
mu = np.empty((K, M-1))
sigma = np.empty((K, M-1))
k = 0
for train_index1, test_index1 in CV.split(X_without_area,y_area):
    
    # extract training and test set for current q CV fold
    x_train1 = X_without_area[train_index1]
    y_train1 = y_area[train_index1]
    x_test1 = X_without_area[test_index1]
    y_test1 = y_area[test_index1]

    D_train1 = (x_train1, y_train1)
    D_test1 = (x_test1, y_test1)
    f = 0
    for train_index2, test_index2 in CV2.split(x_train1, y_train1):
        
        # extract training and test set for current q CV fold
        x_train2 = x_train1[train_index2]
        y_train2 = y_train1[train_index2]
        x_test2 = x_train1[test_index2]
        y_test2 = y_train1[test_index2]
    
        D_train2 = (x_train2, y_train2)
        D_test2 = (x_test2, y_test2)
        
        
        for C in range(0,len(rlr_c)):
            w[:,f,C], test_error[f,C], train_error[f,C] = train_eval_regularized_reg(D_train2, D_test2, rlr_c[C])
    
        
        f=f+1
        
    train_err_vs_lambda = np.mean(train_error,axis=0)
    test_err_vs_lambda = np.mean(test_error,axis=0)
    mean_w_vs_lambda = np.squeeze(np.mean(w,axis=1))
    opt_lambda = rlr_c[np.argmin(np.mean(test_error,axis=0))]
    
    w_rlr[:,k], E_Test, E_Train = train_eval_regularized_reg(D_train1, D_test1, opt_lambda)
    
    Error_test[k] = E_Test / y_test1.shape[0]
                                
    Error_test[k] = E_Test / y_train1.shape[0]
    
    k += 1
    
    if k == K-1:
        figure(k, figsize=(12,8))
        subplot(1,2,1)
        semilogx(rlr_c,mean_w_vs_lambda.T[:,1:],'.--') # Don't plot the bias term
        xlabel('Regularization factor')
        ylabel('Mean Coefficient Values')
        grid()
        # You can choose to display the legend, but it's omitted for a cleaner 
        # plot, since there are many attributes
        #legend(attributeNames[1:], loc='best')
        
        subplot(1,2,2)
        title('Optimal lambda: 1e{0}'.format(np.log10(opt_lambda)))
        loglog(rlr_c,train_err_vs_lambda.T,'b.-',rlr_c,test_err_vs_lambda.T,'r.--')
        xlabel('Regularization factor')
        ylabel('Squared error (crossvalidation)')
        legend(['Train error','Validation error'])
        grid()