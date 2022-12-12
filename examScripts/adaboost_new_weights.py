import numpy as np

### If specific dataset is known
true = [1, 0, 1, 0, 0, 1, 1]
false = np.abs(np.subtract(true,1))
N = np.size(true)

### If size of dataset and percentage is known
# N = 572
# accuracy = 3/4
# true = np.concatenate((np.ones(int(N * accuracy)), np.zeros(int(N*(1-accuracy)))))
# false = np.abs(np.subtract(true,1))


### Adaboost algorithm ###

wi = 1/N                                        # Initial weights
e_t = np.sum(np.multiply(false, wi))            # Errors of classifier
a_t = 1/2 * np.log((1 - e_t) / e_t)

new_weight_correct = true * wi * np.exp(-a_t)
new_weight_wrong = false * wi * np.exp(a_t)

singleWeight = np.add(new_weight_correct, new_weight_wrong)

updatedWeights = singleWeight / np.sum(singleWeight)
print("New weights: ", updatedWeights)

uniqueWeights = np.unique(updatedWeights)
print("Unique weights: ", uniqueWeights)