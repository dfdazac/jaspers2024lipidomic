import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class Min5Imputer(BaseEstimator, TransformerMixin):
    """
    Impute missing values in each feature with the minimum non-missing value divided by 5.
    Works for pandas DataFrames and numpy arrays.
    """
    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.min5_ = (X.min(skipna=True) / 5).to_dict()
        else:
            # Assume numpy array
            self.min5_ = {i: np.nanmin(X[:, i]) / 5 for i in range(X.shape[1])}
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X_filled = X.copy()
            for col in X.columns:
                X_filled[col] = X_filled[col].fillna(self.min5_[col])
            return X_filled
        else:
            # Assume numpy array
            X_filled = np.array(X, copy=True)
            for i in range(X.shape[1]):
                mask = np.isnan(X_filled[:, i])
                X_filled[mask, i] = self.min5_[i]
            return X_filled 