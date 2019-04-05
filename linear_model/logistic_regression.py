from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

from utils import *


def sigmoid(X):
    return 1 / (1 + (np.exp(-X)))

# E = - SUM{y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred)}
def cross_entropy_slow(y_true, y_pred):
    E = 0
    for i in range(len(y_true)):
        if y_true[i] == 1:
            E -= np.log(y_pred[i])
        else:
            E -= np.log(1 - y_pred[i])
    return E

def cross_entropy_faster(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def cross_entropy_fastest(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred))

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)
            


    
class LogisticRegression(object):
    def __init__(self, learning_rate=0.001, epochs=10000, plot_costs=False):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.costs = []
        self.plot_costs = plot_costs
        
    def fit(self, X, y):
        # initialize weights
        self.W = np.random.randn(X.shape[1])
        self.b = 0
        
        for i in range(self.epochs):
            # forward pass
            y_hat = self.predict(X)
            # loss
            loss = cross_entropy_faster(y, y_hat)
            # append to cost list
            self.costs.append(loss)
            # gradient descent
            delta = (y_hat - y)
            self.W -= self.learning_rate * X.T.dot(delta)
            self.b -= self.learning_rate * delta.sum()
            
        if self.plot_costs:
            plt.plot(self.costs)
            plt.show()
          
    def predict(self, X, return_probs=True):
        y_pred = sigmoid(X.dot(self.W) + self.b)
        if not return_probs: y_pred = np.round(y_pred)
        return y_pred