from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import euclidean_distances
import numpy as np
import gzip


class KNNGzip(BaseEstimator, ClassifierMixin):
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        return self

    def predict(self, X):
        y_pred = [self._predict(doc) for doc in X]
        return np.array(y_pred)

    def _predict(self, doc):
        Cx1 = len(gzip.compress(doc.encode()))
        distance_from_x1 = []

        for idx, x2 in zip(self.X_train.index, self.X_train.array):
            Cx2 = len(gzip.compress(x2.encode()))
            x1x2 = " ".join([doc, x2])
            Cx1x2 = len(gzip.compress(x1x2 .encode()))
            ncd = (Cx1x2 - min(Cx1, Cx2)) / max(Cx1, Cx2)
            distance_from_x1.append([idx, ncd])
        
        np_dist = np.array(distance_from_x1)
        chosen = np.argsort(np_dist[:,1])[:self.k]
        k_nearest_labels = self.y_train[np_dist[chosen][:,0]]
        
        unique, pos = np.unique(k_nearest_labels, return_inverse=True)
        counts = np.bincount(pos)
        most_common = np.argmax(counts)
        return unique[most_common]
