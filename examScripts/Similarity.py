# -*- coding: utf-8 -*-
import numpy as np
from toolbox_02450.similarity import similarity

x = np.array([1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1])
y = np.array([1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0])

# Similarity: 'SMC', 'Jaccard', 'ExtendedJaccard', 'Cosine', 'Correlation' 
similarity_measure = 'SMC'
sim1 = similarity(x, y, similarity_measure) 

similarity_measure = 'Jaccard'
sim2 = similarity(x, y, similarity_measure)

similarity_measure = 'Cosine'
sim3 = similarity(x, y, similarity_measure)