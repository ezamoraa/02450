import numpy as np


def density(K, distanceNearestNeighbors):
    density = 1 / (1/K * np.sum(distanceNearestNeighbors, axis=-1))
    return density

def ard(K, densityObservation, densityNearestOtherObservations):
    ard = densityObservation / (1/K * np.sum(densityNearestOtherObservations))
    return ard


# Distances to nearest neighbors for the desired point (observation 1)
distNearestNeighbors1 = [1.04, 1.88] 
K = np.size(distNearestNeighbors1)
density1 = density(K, distNearestNeighbors1)
print("Density for observation: ", density1)



# Distances to nearest neighbors for the K points closest to our point...
# Their NN is not necessarily our initial point!!
distOthers = [
    [0.63, 1.02],       # K nearest neighbors distance for observation 2
    [1.04, 1.80]         # K nearest neighbors distance for observation 3
    ]

densityOthers = density(K, distOthers)
ard1 = ard(K, density1, densityOthers)
print("ARD for observation: ", ard1)


# Check to see if inputs are OK
if K != np.size(distOthers, axis=0) or K != np.size(distOthers, axis=1): print("K does no match with nearest neighbor observations... look through the inputs")