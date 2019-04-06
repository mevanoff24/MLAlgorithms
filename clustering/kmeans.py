import numpy as np
from copy import deepcopy
from utils import *


class KMeans(object):
    def __init__(self, K, max_iter=300, tol=0.0001):
        self.K = K
        self.max_iter = max_iter
        self.tol = tol
        
    def fit(self, X):
        
        N, D = X.shape
        M = np.zeros((self.K, D))
        # randomly initialize K centers
        for k in range(self.K):
            M[k] = X[np.random.choice(N)]
        
        old_clusters = np.zeros(M.shape)
        clusters = np.zeros(len(X))
        i = 0
        error = euclidean_distance(M, old_clusters, axis=None)
        # run algorithm while iteration is less than max iterations or error is greater than tolerance
        while i <= self.max_iter or error >= self.tol:
            for x in range(len(X)):
                # compute distance to all centroids
                dist = euclidean_distance(X[x], M)
                # assign data point to nearest cluster
                clusters[x] = np.argmin(dist)
            
            # to compare error for stopping criteria 
            old_clusters = deepcopy(M)
            
            # assign new centroid by taking the average of all the points assigned to that cluster
            for k in range(self.K):
                assignments = [X[j] for j in range(len(X)) if clusters[j] == k]
                M[k] = np.mean(assignments, axis=0)
            error = euclidean_distance(M, old_clusters, axis=None)
            i += 1
        return clusters