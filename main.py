from datetime import datetime
import json
from typing import Any, Dict
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import f1_score, make_scorer

from sklearn.pipeline import Pipeline

from datasets.sequence_of_words_dataset import SequenceOfWordsDatasetRepository
from pipelines import build_tests
import pandas as pd

def run_test(pipeline_name: str, pipeline: Pipeline, search_params: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
    print(f'Running pipeline \'{pipeline_name}\' with params {search_params} on dataset {dataset_name}')
    search = GridSearchCV(pipeline, search_params, cv=5, refit=True, scoring='f1_micro')

    # train_x, test_x, train_y, test_y = train_test_split(df['Text'], df['Class'], test_size=0.2, random_state = 42)
    # search.fit(train_x, train_y)

    if 'Text' in df.columns:
        search.fit(df['Text'], df['Class'])
    else:
        search.fit(df['text'], df['class'])

    return {
        'best_score': search.best_score_,
        'best_params': search.best_params_,
    }

if __name__ == "__main__":    
    tests = build_tests()

    seq_of_words_repo = SequenceOfWordsDatasetRepository()

    run_complete_texts = False
    for dataset_name in seq_of_words_repo.list_datasets_by_name():
        df = seq_of_words_repo.get_complete_text_dataframe(dataset_name) if run_complete_texts else seq_of_words_repo.get_dataframe(dataset_name)

        for name, pipeline, params in tests:
            result = run_test(name, pipeline, params, df)

            print(f'{result} ---> writing to JSON')

            with open(f'{dataset_name}__{name}__{datetime.now()}.json', 'w') as f:
                f.writelines(json.dumps(result))

