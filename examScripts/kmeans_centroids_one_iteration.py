import numpy as np


data = [-2.1, -1.7, -1.5, -0.4, 0, 0.6, 0.8, 1, 1.1]

centroid_1_init = -2.1
centroid_2_init = -1.7
centroid_3_init = -1.5

dist_1 = np.absolute(np.subtract(data, centroid_1_init))
dist_2 = np.absolute(np.subtract(data, centroid_2_init))
dist_3 = np.absolute(np.subtract(data, centroid_3_init))

print("Dist 1: ", dist_1)
print("Dist 2: ", dist_2)
print("Dist 2: ", dist_3)






centroid_1_iteration_1  = -2.1
centroid_2_iteration_1  = -1.7
centroid_3_iteration_1 = -1.5

second_dist_1 = np.absolute(np.subtract(data, centroid_1_iteration_1))
second_dist_2 = np.absolute(np.subtract(data, centroid_2_iteration_1))
second_dist_3 = np.absolute(np.subtract(data, centroid_3_iteration_1))

print("Second iteration")
print("Dist 1: ", second_dist_1)
print("Dist 2: ", second_dist_2)
print("Dist 2: ", second_dist_3)