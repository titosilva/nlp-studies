from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import euclidean_distances
import numpy as np

class CustomKNN(BaseEstimator, ClassifierMixin):
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        return self

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        distances = euclidean_distances(self.X_train, x)
        sorted_indices = np.argsort(distances, axis = 0)
        k_indices = sorted_indices[:self.k].flatten()
        k_nearest_labels = np.array([self.y_train.values[i] for i in k_indices])
        unique, pos = np.unique(k_nearest_labels, return_inverse=True)
        counts = np.bincount(pos)
        most_common = np.argmax(counts)
        return unique[most_common]