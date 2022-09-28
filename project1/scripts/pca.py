#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 09:35:57 2022

@author: ezamoraa
"""

from data import *
from matplotlib.pyplot import (figure, subplot, plot, xlabel, ylabel, title, 
yticks, show,legend,imshow,grid,xlim,ylim,arrow,text, cm)
import scipy.linalg as linalg

# Center the data (subtract mean column values) and divide by the 
# attribute standard deviation to obtain a standardized dataset
Xc = X - np.ones((N,1))*X.mean(0)
Xc = Xc*(1/np.std(Xc,0)) # we could comment this and show the difference in the plots

# PCA by computing SVD
U,S,V = linalg.svd(Xc,full_matrices=False)
V = V.T

# Compute variance explained by principal components
rho = (S*S) / (S*S).sum() 

# Project data onto principal component space
Z = Xc @ V

 # Plot cumulative variance explained
threshold = 0.9
figure()
plot(range(1,len(rho)+1),rho,'x-')
plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
plot([1,len(rho)],[threshold, threshold],'k--')
title('Variance explained by principal components');
xlabel('Principal component');
ylabel('Variance explained');
legend(['Individual','Cumulative','Threshold'])
grid()
title('Variance explained by principal components')

# PCs to plot
i, j = 0, 1

# Plot attribute coefficients in principal component space
figure()
for att in range(V.shape[1]):
    arrow(0,0, V[att,i], V[att,j])
    text(V[att,i], V[att,j], attributeNames[att])
    
xlim([-1,1])
ylim([-1,1])
xlabel('PC'+str(i+1))
ylabel('PC'+str(j+1))
grid()
# Add a unit circle
plot(np.cos(np.arange(0, 2*np.pi, 0.01)), 
np.sin(np.arange(0, 2*np.pi, 0.01)));
title('Attribute coefficients for PCs')

# Plot PCA of the data (PC1 and PC2)
f = figure()
title('Rice features projected on PCs')
for c in classDict.values():
    # select indices belonging to class c
    class_mask = (y == c)
    plot(Z[class_mask,i], Z[class_mask,j], '.')
legend(classNames)
xlabel('PC1')
ylabel('PC2')
