import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List,Dict

@dataclass
class ValidationReport:
    n_rows: int = 0
    n_cols: int = 0
    errors: List[str] = field(default_factory = list)
    warnings: List[str] = field(default_factory = list)
    is_valid: bool = True

class DataValidator:
    def validate(self, df:pd.DataFrame,target_col:str) -> ValidationReport:
        report = ValidationReport(n_rows=df.shape[0],n_cols=df.shape[1])
        self._check_target_exists(df,target_col,report)
        self._check_duplicate_rows(df,report)
        self._check_constant_columns(df,target_col,report)
        self._check_high_cardinality(df,report)
        self._check_data_types(df,report)

        report.is_valid = len(report.errors) == 0
        return report
    def _check_target_exists(self,df,target_col,report):
        if target_col not in df.columns:
            report.errors.append(
                f'Target column {target_col} not found.' 
                f'Available columns : {list(df.columns)}'
            )
    def _check_duplicate_rows(self, df, report):
        n = df.duplicated().sum()
        if n > 0:
            pct = 100 * n/ len(df)
            report.warnings.append(f'{n} duplicate rows found ({pct:.1f}%). Will be removed.')
 
    def _check_constant_columns(self, df, target_col, report):
        constant_cols = [
            col for col in df.columns
            if col != target_col and df[col].nunique() <= 1
        ]
        if constant_cols:
            report.warnings.append(f'Constant (zero-variance) columns found: {constant_cols}. Will be dropped.')
 
    def _check_high_cardinality(self, df, report, threshold=50):
        cat_cols = df.select_dtypes(include='object').columns
        for col in cat_cols:
            n_unique = df[col].nunique()
            if n_unique > threshold:
                report.warnings.append(
                    f'High cardinality: "{col}" has {n_unique} unique values. '
                    f'Will use TargetEncoder instead of OneHotEncoder.'
                )
 
    def _check_data_types(self, df, report):
        for col in df.select_dtypes(include='object').columns:
            converted = pd.to_numeric(df[col].dropna().head(100), errors='coerce')
            success_rate = converted.notna().mean()
            if success_rate > 0.9:  # >90% of sampled values convert cleanly
                report.warnings.append(
                    f'Column "{col}" appears numeric but stored as string. '
                    f'Will attempt type conversion.'
                )
