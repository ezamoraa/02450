import numpy as np


def index_regression(indexType, y):
    N = np.sum(y)
    mean = np.mean(y)
    p = y / N
    
    if indexType == 'Regression':
        I = 1 / N * np.sum(np.power((y - mean),2))
        
    if indexType == 'Gini':
        I = 1 - np.sum(np.square(p))
    
    if indexType == 'Entropy':
        I = - np.sum(p * np.log2(p))
        
    if indexType == 'ClassError':
        I = 1 - np.max(p)
    
    return N, I



### Regression tree ###
# Initial root [amount of different classes] (if not given it is the sum from the splits)
y0 = [166, 187, 172]

# Split side one
y1 = [108, 112, 56]

# Split side two
y2 = [58, 75, 116]


### Impurity ###
indexType = 'ClassError'

N0, I0 = index_regression(indexType, y0)
N1, I1 = index_regression(indexType, y1)
N2, I2 = index_regression(indexType, y2)


DeltaPurityGain = I0 - N1 / N0 * I1 - N2 / N0 * I2
print("Purity gain: ", DeltaPurityGain)