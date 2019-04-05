from sklearn.datasets import make_classification
import numpy as np

def get_simple_classification_data(random_state=102):
    X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=1, 
                                                n_clusters_per_class=1, random_state=random_state)
    return X, y

    

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


    
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

