from typing import List
from datasets.dataframe_repository import DataframeRepository
import pandas as pd

class SequenceOfWordsDatasetRepository:
    def __init__(self) -> None:
        self.__repository = DataframeRepository()

    def get_dataframe(self, dataset_name: str) -> pd.DataFrame:
        return self.__repository.get_data_from_source(self.get_fetch_url(dataset_name), compression='zip')
    
    def get_complete_text_dataframe(self, dataset_name: str) -> pd.DataFrame:
        return self.__repository.get_data_from_source(self.get_complete_text_fetch_url(dataset_name))
    
    def get_fetch_url(self, dataset_name: str) -> str:
        return f'https://github.com/ragero/text-collections/raw/master/Sequence_of_words_CSV/{dataset_name}.csv.zip'
    
    def get_complete_text_fetch_url(self, dataset_name: str) -> str:
        return f'https://github.com/ragero/text-collections/raw/master/complete_texts_csvs/{dataset_name}.csv'

    def list_datasets_by_name(self) -> List[str]:
        return [
            # 'CSTR', 
            # 'classic4',
            # 're8',
            'Dmoz-Sports',
            'Dmoz-Health',
            'Dmoz-Science',
        ]