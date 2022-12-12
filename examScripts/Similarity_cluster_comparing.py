import numpy as np
from sklearn import metrics


def numberPairs(n):
    numberPairs = np.sum((n * (n - 1)) / 2)
    return numberPairs

def rand(S, D, N):
    rand = (S + D) / (1/2 * N * (N - 1))
    return rand

def jaccard(S, D, N):
    jaccard = S / (1/2 * N * (N - 1) - D)
    return jaccard

def statsFromMatrix(n):
    #sizeVertical = np.size(n, 0)                # Eq to amount of rows
    #sizeHorizontal = np.size(n, 1)              # Eq to amount of columns
    totalValues = np.sum(n)
    totalPossiblePairs = (totalValues * (totalValues - 1)) / 2
    sumVertical = np.sum(n, axis = 1)           # Sum of each row
    sumHorizontal = np.sum(n, axis = 0)         # Sum of each column
    return totalValues, totalPossiblePairs, sumVertical, sumHorizontal

def entropy(n, N):
    p = 1 / N * n
    p = p[p != 0]
    H = - np.sum(p * np.log(p)) 
    return H
    


# Clusters
C1 = np.array([1, 1, 1, 1, 2, 2, 3, 3, 3])
C2 = np.array([4, 4, 1, 1, 2, 2, 2, 3, 3])

C1 = np.array([1, 1, 1, 2, 2, 2, 2, 2, 3, 3])
C2 = np.array([2, 3, 2, 1, 2, 1, 2, 2, 2, 2])

clusterNames1 = np.unique(C1)
clusterNames2 = np.unique(C2)



### confusion matrix (or counting matrix) ###
n = np.zeros((np.size(np.unique(C1)), np.size(np.unique(C2))))
for i_C1, n_C1 in enumerate(n):
    for i_C2, __ in enumerate(n_C1):
        compareClusters = np.logical_and(C1 == clusterNames1[i_C1], C2 == clusterNames2[i_C2])
        n[i_C1,i_C2] = np.sum(compareClusters)

# n = metrics.confusion_matrix(C1, C2)                      # Calculate by sklearn
# n = np.array([[114, 0, 32], [0, 119, 0], [8, 0, 60]])     # If you have a given confusion matrix
print("Confusion matrix: ")
print(n)



# Calculate from matrix
N, total_pairs, nC1, nC2  = statsFromMatrix(n)

### Calculate S ###
S = numberPairs(n)
print("S: ", S)

### Calculate D ###
D = total_pairs - numberPairs(nC1) - numberPairs(nC2) + S
print("D: ", D)

        

### Rand index (SMC) ###
rand = rand(S,D,N)
print("Rand: ", rand)

### Jaccard ###
jaccard = jaccard(S,D,N)
print("Jaccard: ", jaccard)


### Normalized Mutual Information ###
HC1 = entropy(nC1, N)
HC2 = entropy(nC2, N)
HC1C2 = entropy(n, N)

MI = HC1 + HC2 - HC1C2
NMI = MI / (np.sqrt(HC1) * np.sqrt(HC2))
print("MI: ", MI)
print("NMI: ", NMI)