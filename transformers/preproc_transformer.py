from sklearn.base import BaseEstimator, TransformerMixin
from nltk import PorterStemmer

class PreprocTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.stemmer = PorterStemmer()
        return self

    def transform(self, X, y=None):
        X.apply(lambda x: self.stemmer.stem(x))
        return X