import numpy as np
from sklearn.linear_model import LinearRegression


# X = np.array([1, 2, 3, 4])
# X = np.transpose(np.array([np.cos(np.pi / 2 * X), np.sin(np.pi / 2 * X)]))

X = np.transpose(np.array([[1, 2, 3, 4],[2,3,4,5]]))
y = np.reshape(np.array([6, 2, 3, 4]), (-1,1))


reg = LinearRegression().fit(X, y)
score = reg.score(X, y)
coef_ = reg.coef_
intercept_ = reg.intercept_

### Prediction for unseen values (xPredict)
xPredict = np.transpose(np.array([[5],[2]]))
yhat = reg.predict(xPredict)