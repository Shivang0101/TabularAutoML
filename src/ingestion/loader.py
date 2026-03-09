import openml
import pandas as pd
from pathlib import Path
import logging

logger= logging.getLogger(__name__)

class DataLoader:
    def load_csv(self,path:str) -> pd.DataFrame:
        path=Path(path)
        if not path.exists():
            raise FileNotFoundError(f'csv not found at {path}')
        logger.info(f'Loading csv file from {path}')
        df=pd.read_csv(path)
        logger.info(f'Loaded {df.shape[0]} rows and {df.shape[1]} columns.')
        return self._basic_validate(df)
    
    def load_openml(self,dataset_id: int) -> pd.DataFrame:
        logger.info(f"Fetching OpenML datset with id = {dataset_id} .")
        df=openml.datasets.get_dataset(dataset_id)
        X,y,_,_ = datset.get_data(target=dataset.default_target_attribute)
        df1 = pd.concat([X,y.rename('target')],axis=1)
        logger.info(f'OpenML dataset loaded: {df.shape}')
        return self._basic_validate(df1)
    
    def basic_validate_(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.shape[0] < 30 : 
            raise ValueError(f'Dataset is too small : {df.shape[0]} rows. Need atleast 30.')
        if df.shape[1] < 2 :
            raise ValueError(f'Dataset has only {df.shape[1]} columns . Need atleast 2. ')
        missing_points=df.isnull().sum()
        alert=missing_points[missing_points > 0.5].index.tolist()
        if alert:
            logger.warning(f'High missing data ( >50%) in columns : {alert} .')
        return df