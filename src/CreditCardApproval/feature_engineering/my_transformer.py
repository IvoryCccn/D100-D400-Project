from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


@dataclass
class Log1pTransformer(BaseEstimator, TransformerMixin):
    """
    Apply log(1 + x) transformation to selected numeric columns.

    Notes:
        - Intended for non-negative variables (e.g., income, counts, durations).
        - If negative values are found during transform, an error is raised to avoid silently producing NaNs.
    """

    columns: Optional[Iterable[str]] = None
    add_suffix: bool = True
    suffix: str = "_log1p"
    
    def fit(self, X: pd.DataFrame, y=None):
        """
        Record the column names requiring transformation.
        """
        X = self._check_X(X)
        self.columns_ = list(self.columns) if self.columns is not None else X.columns.tolist()
        self._validate_columns_exist(X, self.columns_)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Perform the log(1 + x) transformation on the specified columns and return the new DataFrame.
        """
        X = self._check_X(X).copy()
        self._validate_columns_exist(X, self.columns_)

        for col in self.columns_:
            vals = X[col].astype(float)

            # allow NaN, but disallow negative numbers
            if np.nanmin(vals.values) < 0:
                raise ValueError(
                    f"Log1pTransformer: column '{col}' contains negative values. "
                    "log1p is defined for x >= -1, but negative income/duration is unexpected."
                )

            new_col = f"{col}{self.suffix}" if self.add_suffix else col
            X[new_col] = np.log1p(vals)

        return X

    def get_feature_names_out(self, input_features=None):
        """
        Returns a list of transformed feature names, compatible with the sklearn pipeline interface.
        """
        if input_features is None:
            input_features = self.columns_
        input_features = list(input_features)

        out = input_features.copy()
        for col in self.columns_:
            new_col = f"{col}{self.suffix}" if self.add_suffix else col
            if new_col not in out:
                out.append(new_col)
        return np.array(out, dtype=object)

    @staticmethod
    def _check_X(X):
        """
        Verify whether the input is a pandas DataFrame.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Log1pTransformer expects a pandas DataFrame as input.")
        return X

    @staticmethod
    def _validate_columns_exist(X: pd.DataFrame, cols: List[str]) -> None:
        """
        Verify whether all specified columns are present in the DataFrame.
        """
        missing = [c for c in cols if c not in X.columns]
        if missing:
            raise ValueError(f"Log1pTransformer: missing columns {missing}.")
            