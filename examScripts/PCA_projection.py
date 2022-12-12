import numpy as np

# Principal component vector
V1 = np.array([0.45, -0.4, 0.58, 0.55])
V2 = np.array([0.6, 0.2, -0.08, -0.3])

# New point to project
x = np.array([-1, -1, -1, 1])

# Projection on PC
b1 = np.sum(np.transpose(x) * V1)
b2 = np.sum(np.transpose(x) * V2)

print("Projection to PC1", b1)
print("Projection to PC2", b2)




V = np.array([V1, V2])
b = np.matmul(V, x)
print("projection: ", b)