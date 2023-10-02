from typing import Any, Dict, List, Tuple
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import GridSearchCV

from classifiers.knn import CustomKNN
from classifiers.knn_gzip import KNNGzip
from classifiers.naive_bayes import CustomNaiveBayes
from transformers.preproc_transformer import PreprocTransformer

def build_tests() -> List[Tuple[str, Pipeline, Dict[str, Any]]]:
    knn_pipeline = Pipeline(steps=[
        ('vectorize', TfidfVectorizer()),
        ('clf', CustomKNN())
    ])

    nb_pipeline = Pipeline(steps=[
        ('vectorize', CountVectorizer()),
        ('clf', CustomNaiveBayes())
    ])

    knn_gzip_pipeline = Pipeline(steps=[
        ('clf', KNNGzip())
    ])

    return [
        ('knn', knn_pipeline, {'clf__k': [1, 15]}),
        ('nb', nb_pipeline, {}),
        ('knn_gzip', knn_gzip_pipeline, {'clf__k': [1, 15]}),
    ]

