from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

class CustomNaiveBayes(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.class_probs = None
        self.feature_probs = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        
        # Calculate class probabilities
        self.class_probs = {
            label: np.mean(y == label) for label in np.unique(y)
        }

        # Calculate feature probabilities
        self.feature_probs = {
            label: (np.sum(X[y == label], axis=0) + 1) / (np.sum(X[y == label]) + X.shape[1])
            for label in np.unique(y)
        }

    def predict_proba(self, X):
        log_probs = {}

        # Calculate log probabilities
        for label in self.class_probs:
            class_prob_log = np.log(self.class_probs[label])
            x_ar = X.toarray().flatten()
            feature_probs_ar = np.squeeze(np.asarray(self.feature_probs[label].flatten()))
            feature_probs_log = np.sum(np.log(feature_probs_ar) * x_ar)

            log_probs[label] = class_prob_log + feature_probs_log

        return log_probs

    def predict(self, X):
        predicts = []
        for x in X:
            probs = self.predict_proba(x)
            predicts.append(np.array([max(probs, key=probs.get)]))

        return np.array(predicts)