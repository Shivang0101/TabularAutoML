# ─────────────────────────────────────────────────────────
# PURPOSE: Remove outliers.
#
# TWO METHODS applied in sequence (configurable via `method` param):
#   1. IQR Method:  Statistical rule. Any value outside
#                   [Q1 - 1.5*IQR, Q3 + 1.5*IQR] is an outlier.
#                   Fast, interpretable, works column-by-column.
#   2. Isolation Forest: ML-based anomaly detection.
#                   Randomly isolates points — outliers are isolated faster.
#                   Catches multivariate outliers the IQR method misses.
#                   E.g., age=95, salary=20k might each be 'normal' alone
#                   but together are an unusual combination.
#
# METHOD OPTIONS:
#   'both'             — IQR first, then IsolationForest (default)
#   'iqr'              — IQR only
#   'isolation_forest' — IsolationForest only
#   'none'             — skip outlier removal entirely (already clean data)
#
# SAFETY GUARD:
#   If IQR alone removes > max_removal_pct of rows (default 20%),
#   IsolationForest is skipped automatically to prevent over-filtering.
# ─────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import logging

logger = logging.getLogger(__name__)


class OutlierHandler:

    def __init__(self, iqr_multiplier: float = 1.5,
                 if_contamination: float = 0.05,
                 method: str = 'isolation_forest',
                 max_removal_pct: float = 0.10):
        self.iqr_multiplier = iqr_multiplier        # how far beyond IQR is an outlier
        self.if_contamination = if_contamination    # % of data IsolationForest treats as outliers
        self.method = method                        # 'both', 'iqr', 'isolation_forest', 'none'
        self.max_removal_pct = max_removal_pct      # safety cap — stop if too many rows removed

    def remove_outliers(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        if self.method == 'none':
            logger.info('Outlier removal skipped (method=none).')
            return df

        original_len = len(df)

        if self.method in ('iqr', 'both'):
            df = self._iqr_filter(df, target_col)
            removed_after_iqr = original_len - len(df)
            removed_pct = removed_after_iqr / original_len
            logger.info(
                f'After IQR filter: {len(df)} rows '
                f'(removed {removed_after_iqr} = {removed_pct:.1%})'
            )
            if self.method == 'both' and removed_pct > self.max_removal_pct:
                logger.warning(
                    f'[OutlierHandler] IQR removed {removed_pct:.1%} of rows '
                    f'(threshold: {self.max_removal_pct:.1%}) — '
                    f'skipping IsolationForest to avoid over-filtering. '
                    f'Consider raising iqr_multiplier (currently {self.iqr_multiplier}).'
                )
                return df  
        if self.method in ('isolation_forest', 'both'):
            df = self._isolation_forest_filter(df, target_col)
            logger.info(
                f'After IsolationForest: {len(df)} rows '
                f'(removed {original_len - len(df)} total)'
            )

        return df

    def _iqr_filter(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [c for c in numeric_cols if c != target_col]

        mask = pd.Series([True] * len(df), index=df.index)

        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            lower = Q1 - self.iqr_multiplier * IQR
            upper = Q3 + self.iqr_multiplier * IQR

            col_mask = df[col].between(lower, upper)
            mask = mask & col_mask  

        return df[mask].reset_index(drop=True)

    def _isolation_forest_filter(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                        if c != target_col]

        if len(feature_cols) < 2:
            logger.info('IsolationForest skipped — need at least 2 numeric columns.')
            return df

        iso = IsolationForest(
            contamination=self.if_contamination,
            random_state=42,    # reproducibility — same result every run
            n_estimators=100    # number of trees in the forest
        )
        predictions = iso.fit_predict(df[feature_cols])
        normal_mask = predictions == 1  # 1 = normal, -1 = outlier
        return df[normal_mask].reset_index(drop=True)