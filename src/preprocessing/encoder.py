# ─────────────────────────────────────────────────────────
# encoder.py — Categorical Feature Encoding
# PURPOSE: Convert text categories to numbers for ML models.
#
# FOUR STRATEGIES (auto-selected by cardinality):
#   DROP            — junk cardinality  (> 90% unique values)
#                     e.g., 'customer_id': every row unique → useless, dropped
#   OneHotEncoder   — low cardinality  (<= 15 unique values)
#                     e.g., 'color': [Red, Blue, Green] → 3 columns
#   OrdinalEncoder  — high cardinality (> 15 unique values)
#                     e.g., 'city': [NY, LA, Chicago, ...] → 1 column of integers
#   TargetEncoder   — very high cardinality (user can specify explicitly)
#                     Replaces category with mean of target for that category.
#                     e.g., 'zipcode' → average house price for that zip code
#
# CARDINALITY THRESHOLDS:
#   > 90% unique  → DROP   (IDs, hashes, UUIDs — no learnable pattern)
#   50–90% unique → WARN   (risky but kept, target-encoded)
#   16–50% unique → ORDINAL
#   <= 15 unique  → OHE
# ─────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from category_encoders import TargetEncoder
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class AutoEncoder:
    def __init__(self, ohe_threshold: int = 15,
                 target_encode_threshold: int = 50,
                 drop_cardinality_threshold: float = 0.90):
        self.ohe_threshold = ohe_threshold
        self.target_encode_threshold = target_encode_threshold
        self.drop_cardinality_threshold = drop_cardinality_threshold  
        self._encoders: Dict = {}
        self._strategy: Dict = {}
        self._ohe_cols_output: List = []
        self._dropped_cols: List = []   

    def fit_transform(self, df: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
        df = df.copy()
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        for col in cat_cols:
            n_unique = df[col].nunique()
            cardinality_ratio = n_unique / len(df)  
            if cardinality_ratio > self.drop_cardinality_threshold:
                df.drop(columns=[col], inplace=True)
                self._dropped_cols.append(col)
                self._strategy[col] = 'drop'
                logger.warning(
                    f"[AutoEncoder] DROP '{col}': {n_unique}/{len(df)} unique values "
                    f"({cardinality_ratio:.1%} cardinality) — likely an ID column."
                )
                continue 
            elif n_unique <= self.ohe_threshold:
                enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first')
                encoded = enc.fit_transform(df[[col]])
                new_cols = [f'{col}_{cat}' for cat in enc.categories_[0][1:]]
                self._ohe_cols_output.extend(new_cols)
                df = pd.concat([
                    df.drop(columns=[col]),
                    pd.DataFrame(encoded, columns=new_cols, index=df.index)
                ], axis=1)
                self._encoders[col] = enc
                self._strategy[col] = 'ohe'

            elif n_unique <= self.target_encode_threshold:
                enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                df[[col]] = enc.fit_transform(df[[col]])
                self._encoders[col] = enc
                self._strategy[col] = 'ordinal'

            else:
                logger.warning(
                    f"[AutoEncoder] WARN '{col}': {n_unique}/{len(df)} unique values "
                    f"({cardinality_ratio:.1%} cardinality) — target-encoding, but verify "
                    f"this isn't an ID column."
                )
                enc = TargetEncoder(smoothing=10)
                df[[col]] = enc.fit_transform(df[[col]], target)
                self._encoders[col] = enc
                self._strategy[col] = 'target'

        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        cols_to_drop = [col for col in self._dropped_cols if col in df.columns]
        if cols_to_drop:
            df.drop(columns=cols_to_drop, inplace=True)

        for col, enc in self._encoders.items():
            if col not in df.columns:
                continue
            strategy = self._strategy[col]
            if strategy == 'ohe':
                encoded = enc.transform(df[[col]])
                new_cols = [f'{col}_{cat}' for cat in enc.categories_[0][1:]]
                df = pd.concat([
                    df.drop(columns=[col]),
                    pd.DataFrame(encoded, columns=new_cols, index=df.index)
                ], axis=1)
            elif strategy in ('ordinal', 'target'):
                df[[col]] = enc.transform(df[[col]])

        return df

    def get_dropped_columns(self) -> List[str]:
        return self._dropped_cols.copy()