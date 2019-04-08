from sklearn.datasets import make_classification
from sklearn.datasets import make_blobs
import numpy as np

"""
Datasets
"""
def get_simple_classification_data(random_state=102):
    X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=1, 
                                                n_clusters_per_class=1, random_state=random_state)
    return X, y

def get_clustering_data(random_state=100, cluster_std=3.0): 
    X, true_clusters = make_blobs(random_state=random_state, cluster_std=cluster_std)
    return X, true_clusters
    

"""
Formulas
"""
def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

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

# np.sqrt([np.sum((i - j)**2 for i, j in zip (a, b))])
def euclidean_distance(a, b, axis=1):
    return np.linalg.norm(a - b, axis=axis)


def entropy(y):
    # assume y is binary - 0 or 1
    N = len(y)
    s1 = (y == 1).sum()
    if 0 == s1 or N == s1:
        return 0
    p1 = float(s1) / N
    p0 = 1 - p1
    return -p0 * np.log2(p0) - p1 * np.log2(p1)

def gini_index(groups, classes):
    # count all samples at split point
    n_instances = float(sum([len(group) for group in groups]))
    # sum weighted Gini index for each group
    gini = 0.0
    for group in groups:
        size = float(len(group))
        # avoid divide by zero
        if size == 0:
            continue
        score = 0.0
        # score the group based on the score for each class
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
            score += p * p
        # weight the group score by its relative size
        gini += (1.0 - score) * (size / n_instances)
    return gini

