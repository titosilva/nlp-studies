import re
from hashlib import sha256
import os
import pandas as pd
import urllib.request

class DataframeRepository:
    __prefetch_folder_abs: str

    def __init__(self) -> None:
        self.__prefetch_folder_abs = "/tmp/dataset_prefetch"

    def get_data_from_source(self, data_src: str, compression: str = 'infer') -> pd.DataFrame:
        urls = re.findall(r'(https?://[^\s]+)', data_src)

        if len(urls) == 0:
            return self.__read_data_from_file(data_src)
        
        prefetch_path = self.__build_prefetch_filepath(urls[0])

        if self.__detect_prefetch(prefetch_path):
            print(f'Reading {data_src} from prefetch located in {prefetch_path}')
            return self.__read_data_from_file(prefetch_path, compression=compression)
        
        return self.__fetch_then_read(urls[0], prefetch_path, compression=compression)

    def __read_data_from_file(self, path: str, compression: str) -> pd.DataFrame:
        return pd.read_csv(path, compression=compression)
        
    def __detect_prefetch(self, prefetch_path: str) -> bool:
        return os.path.exists(prefetch_path)
    
    def __build_prefetch_filepath(self, url: str):
        filename = sha256(url.encode()).hexdigest()
        return os.path.join(self.__prefetch_folder_abs, filename)

    def __fetch_then_read(self, url: str, path: str, compression: str) -> pd.DataFrame:
        if path is None:
            raise TypeError(f'Path cannot be None')
        
        print(f'Fetching data from {url} and writing to {path}')
        os.makedirs(os.path.split(path)[0], exist_ok=True)
        urllib.request.urlretrieve(url, path)

        print(f'Data fetched. Reading...')
        return pd.read_csv(path, compression=compression)