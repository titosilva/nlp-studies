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

    knn_pipeline_pre = Pipeline(steps=[
        ('preproc', PreprocTransformer()),
        ('vectorize', TfidfVectorizer()),
        ('clf', CustomKNN())
    ])

    nb_pipeline_pre = Pipeline(steps=[
        ('preproc', PreprocTransformer()),
        ('vectorize', CountVectorizer()),
        ('clf', CustomNaiveBayes())
    ])

    knn_gzip_pipeline_pre = Pipeline(steps=[
        ('preproc', PreprocTransformer()),
        ('clf', KNNGzip())
    ])

    return [
        # ('knn', knn_pipeline, {'clf__k': [1, 15], 'vectorize__max_df': [0.95, 0.9, 0.85, 0.8]}),
        # ('knn_pre', knn_pipeline_pre, {'clf__k': [1, 15], 'vectorize__max_df': [0.95, 0.9, 0.85, 0.8]}),
        ('nb', nb_pipeline, {'vectorize__max_df': [0.95, 0.9, 0.85, 0.8], 'vectorize__min_df': [0.05, 0.1, 0.15, 0.2]}),
        # ('nb_pre', nb_pipeline_pre, {'vectorize__max_df': [0.95, 0.9, 0.85, 0.8], 'vectorize__min_df': [0.05, 0.1, 0.15, 0.2]}),
        # ('knn_gzip', knn_gzip_pipeline, {'clf__k': [1, 15], 'vectorize__max_df': [0.95, 0.9, 0.85, 0.8]}),
        # ('knn_gzip_re', knn_gzip_pipeline_pre, {'clf__k': [1, 15], 'vectorize__max_df': [0.95, 0.9, 0.85, 0.8]}),
    ]

