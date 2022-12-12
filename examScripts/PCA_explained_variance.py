import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd


S = np.array([43.4, 23.39, 18.26, 9.34, 2.14])

rho = (S*S) / (S*S).sum() 

rhosum = np.cumsum(rho)


threshold = 0.8

# Plot variance explained
plt.figure()
plt.plot(range(1,len(rho)+1),rho,'x-')
plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
plt.plot([1,len(rho)],[threshold, threshold],'k--')
plt.title('Variance explained by principal components');
plt.xlabel('Principal component');
plt.ylabel('Variance explained');
plt.legend(['Individual','Cumulative','Threshold'])
plt.grid()
plt.show()


print("Explained variance: ", rhosum)

# Calculate a part of the variance explained (e.g. principal component 3,4,5)
# rhosumpart = np.cumsum(rho[2:5])
rhosumpart = np.cumsum(rho[1:6])
print("Explained variance from parts: ", rhosumpart)