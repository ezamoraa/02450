# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 09:55:37 2022

@author: Steven
"""
from matplotlib.pylab import figure, subplot, plot, xlabel, ylabel, hist, show
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
from data import *
from sklearn.model_selection import train_test_split


lm_idx = attributeNames.index('Area')
y = X[:, lm_idx]

X_cols = list(range(0, lm_idx)) + list(range(lm_idx+1,len(attributeNames)))
X = X[:, X_cols]
attributeNames.pop(lm_idx)

ca_idx = attributeNames.index('Convex_Area')
X_cols = list(range(0, ca_idx)) + list(range(ca_idx+1,len(attributeNames)))
X = X[:, X_cols]
attributeNames.pop(ca_idx)



X = np.append(X, onehot_encoded, 1)
attributeNames.append("Cameo")
attributeNames.append("Osmancik")


test_proportion = 0.3
X_train1, X_test1, y_train1, y_test1 = train_test_split(X,y,test_size=test_proportion)


# test_proportion = 0.3
# X_train2, X_test2, y_train2, y_test2 = train_test_split(X_test1,y_test1,test_size=test_proportion)

# Fit model to data

mu = np.mean(X_train1[:, 0:], 0)
sigma = np.std(X_train1[:, 0:], 0)

X_train1[:, 0:] = (X_train1[:, 0:] - mu) / sigma
X_test1[:, 0:] = (X_test1[:, 0:] - mu) / sigma


model = lm.LinearRegression(fit_intercept=True)
model.fit(X_train1, y_train1)


# Predict Area

y_est = model.predict(X_test1)
residual = y_test1 - y_est

# Display scatter plot
plt.figure(figsize=(10,10))

axis_range = [np.min([y_est, y_test1]),np.max([y_est, y_test1])]
#subplot(2,1,1)
plot(y_test1, y_est, '.')
plot(axis_range,axis_range,'k--')
xlabel('Area (true)'); ylabel('Area (estimated)');
plt.legend(['Model estimations','Perfect estimation'])
plt.title('Linear regression')
plt.ylim(axis_range); plt.xlim(axis_range)
plt.grid()
#subplot(2,1,2)
#hist(residual,40)

show()

print('Ran Linear regression')

#array([-1.95849708e+04,  1.15653140e+04,  2.09057310e+03,  1.36947035e+00,
    #    2.40612171e+06, -4.95639317e-01])