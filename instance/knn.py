import numpy as np
import heapq
from utils import *



class KNN(object):
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        
    def fit(self, X, y):
        self.X = X
        self.y = y
        
    def predict(self, X):
        
        y = np.zeros(len(X))
        for i, x in enumerate(X):
            heap = []
            heapq.heapify(heap)
            for j, xj in enumerate(self.X):
                diff = x - xj
                distance = diff.dot(diff)
                if len(heap) < self.n_neighbors:
                    heapq.heappush(heap, (-distance, self.y[j]))
                else:
                    heapq.heappushpop(heap, (-distance, self.y[j]))
                    
        
            votes = {}
            for point in heap:
                K = point[1]
                votes[K] = votes.get(K, 0) + 1
            
            max_votes = 0
            max_votes_class = -1
            for v, count in votes.items():
                if count > max_votes:
                    max_votes = count
                    max_votes_class = v
            y[i] = max_votes_class
                
        return y
    
    def score(self, X, y):
        y_hat = self.predict(X)
        return np.mean(y == y_hat)
        