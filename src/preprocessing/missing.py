# ─────────────────────────────────────────────────────────
# missing.py — Intelligent Missing Value Imputation
# PURPOSE: Fill NaN values using context-appropriate strategies.
#          Different column types need different imputation methods.
# STRATEGIES:
#   Numeric + low missing (< 20%): KNN Imputer (uses neighbor values)
#   Numeric + high missing (>= 20%): Median (robust to outliers)
#   Categorical: Mode (most frequent value)
# ─────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer, SimpleImputer

class MissingValueHandler:
    def __init__(self, knn_threshold: float = 0.20,knn_neighbors:int=5):
        self.knn_threshold = knn_threshold
        self.knn_neighbors = knn_neighbors
        self._imputers = {}  
    def fit_transform(self, df: pd.DataFrame)->pd.DataFrame:
        df=df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        for col in numeric_cols:
            missing_points = df[col].isnull().mean()  
            if missing_points == 0:
                continue   
            if missing_points < self.knn_threshold:
                imputer = KNNImputer(n_neighbors=self.knn_neighbors)
                df[[col]] = imputer.fit_transform(df[[col]])
            else:
                imputer = SimpleImputer(strategy='median')
                df[[col]] = imputer.fit_transform(df[[col]])
            self._imputers[col] = imputer
        for col in cat_cols:
            if df[col].isnull().sum() == 0:
                continue
            imputer = SimpleImputer(strategy='most_frequent')
            df[[col]] = imputer.fit_transform(df[[col]])
            self._imputers[col] = imputer
        return df

    def transform(self, df:pd.DataFrame) -> pd.DataFrame:
        df=df.copy()
        for col, imputers in self._imputers.items():
            if col in df.columns:
                df[[col]] = imputers.transform(df[[col]])
        return df

