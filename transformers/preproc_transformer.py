from sklearn.base import BaseEstimator, TransformerMixin
from nltk import PorterStemmer
import nltk

class PreprocTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.stemmer = PorterStemmer()
        self.stopwords = nltk.corpus.stopwords.words('english')
        return self
    
    def __transform(self, text: str):
        tokenized = nltk.word_tokenize(text)
        lower = map(str.lower, tokenized)
        return ' '.join(map(self.stemmer.stem, filter(lambda y: y not in self.stopwords, lower)))


    def transform(self, X, y=None):        
        return X.map(self.__transform)