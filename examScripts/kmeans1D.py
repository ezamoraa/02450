import numpy as np
from sklearn.cluster import KMeans

"""
Try the script on:
2016 fall Q11
2017 fall Q8

"""

X = np.array([-2.1, -1.7, -1.5, -0.4, 0.0, 0.6, 0.8, 1.0, 1,1]).reshape(-1,1)

nClusters = 3
withInitialCluster = True
initial_cluster = np.array([-2.1, -1.7, -1.5]).reshape(-1,1)


if withInitialCluster:
    kmeans = KMeans(n_clusters=nClusters, n_init=1, max_iter=1000, init=initial_cluster).fit(X)
else:
    kmeans = KMeans(n_clusters=nClusters, n_init=100, max_iter=1000).fit(X)


print("Cluster centers = {}".format(kmeans.cluster_centers_))
print("Labels = {}".format(kmeans.labels_))
