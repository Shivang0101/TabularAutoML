# ─────────────────────────────────────────────────────────
# scaler.py — Automatic Feature Scaling
# PURPOSE: Normalize feature ranges so no single feature dominates.
#
# TWO STRATEGIES (auto-selected by distribution skewness):
#   StandardScaler  — for normally distributed features
#                     Transforms to mean=0, std=1
#                     Formula: z = (x - mean) / std
#   RobustScaler    — for skewed distributions / features with outliers
#                     Uses median and IQR instead of mean and std
#                     Formula: z = (x - median) / IQR
#                     WHY: Mean/std are pulled by outliers, median/IQR are not
# ─────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from typing import Dict

class AutoScaler:
    def __init__(self, skew_threshold: float = 1.0):
        self.skew_threshold = skew_threshold
        self._scalers: Dict = {}
 
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns 
        for col in numeric_cols:
            unique_vals = df[col].nunique()
            if unique_vals <= 2:
                continue
            col_skew = df[col].skew()
 
            if abs(col_skew) > self.skew_threshold:
                scaler = RobustScaler()
            else:
                scaler = StandardScaler()
            df[[col]] = scaler.fit_transform(df[[col]])
            self._scalers[col] = scaler
        return df
 
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col, scaler in self._scalers.items():
            if col in df.columns:
                df[[col]] = scaler.transform(df[[col]])
        return df
