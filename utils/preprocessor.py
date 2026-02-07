import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy import stats

class TimeSeriesPreprocessor:
    """Configurable preprocessing pipeline for time series."""

    def __init__(self, df, target_col):
        self.df = df.copy()
        self.target_col = target_col
        self.original_df = df.copy()
        self.preprocessing_log = []

    # ---------- Missing values ----------
    def handle_missing_values(self, method='ffill'):
        """Handle missing values in target column."""
        missing_before = self.df[self.target_col].isnull().sum()

        if method == 'ffill':
            self.df[self.target_col] = self.df[self.target_col].ffill().bfill()
            method_name = "Forward + backward fill"
        elif method == 'bfill':
            self.df[self.target_col] = self.df[self.target_col].bfill().ffill()
            method_name = "Backward + forward fill"
        elif method == 'interpolate':
            self.df[self.target_col] = self.df[self.target_col].interpolate(method='linear').ffill().bfill()
            method_name = "Linear interpolation"
        elif method == 'mean':
            self.df[self.target_col] = self.df[self.target_col].fillna(self.df[self.target_col].mean())
            method_name = "Mean imputation"
        elif method == 'drop':
            self.df = self.df.dropna(subset=[self.target_col])
            method_name = "Row drop"
        else:
            method_name = "No missing handling"

        missing_after = self.df[self.target_col].isnull().sum()
        self.preprocessing_log.append(
            f"Missing values: {missing_before} → {missing_after} using [{method_name}]"
        )
        return self

    # ---------- Outliers ----------
    def handle_outliers(self, method='clip', threshold=3.0):
        """Handle outliers in target column."""
        series = self.df[self.target_col]
        outliers_before = 0
        outliers_after = 0
        method_name = "None"

        if method == 'clip':
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - threshold * IQR
            upper = Q3 + threshold * IQR
            mask = (series < lower) | (series > upper)
            outliers_before = mask.sum()
            self.df[self.target_col] = series.clip(lower, upper)
            method_name = f"IQR clipping (±{threshold} IQR)"
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(series, nan_policy='omit'))
            mask = z_scores > threshold
            outliers_before = np.nansum(mask)
            self.df.loc[mask, self.target_col] = series.median()
            method_name = f"Z-score replacement (|z|>{threshold})"

        self.preprocessing_log.append(
            f"Outliers handled: ~{int(outliers_before)} adjusted using [{method_name}]"
        )
        return self

    # ---------- Scaling / transforms ----------
    def apply_scaling(self, method='none'):
        """Scale or transform target column."""
        series = self.df[self.target_col]

        if method == 'standard':
            scaler = StandardScaler()
            self.df[self.target_col] = scaler.fit_transform(series.values.reshape(-1, 1))
            self.preprocessing_log.append("Applied StandardScaler (mean 0, std 1).")
        elif method == 'minmax':
            scaler = MinMaxScaler()
            self.df[self.target_col] = scaler.fit_transform(series.values.reshape(-1, 1))
            self.preprocessing_log.append("Applied MinMaxScaler (scaled to [0,1]).")
        elif method == 'log':
            shifted = series
            offset = 0
            if (series <= 0).any():
                offset = -series.min() + 1e-6
                shifted = series + offset
            self.df[self.target_col] = np.log(shifted)
            self.preprocessing_log.append(
                f"Applied log-transform (shifted by {offset:.4f} to ensure positivity)."
            )
        else:
            self.preprocessing_log.append("No scaling/transform applied.")
        return self

    # ---------- Differencing ----------
    def apply_differencing(self, order=1):
        """Apply differencing to remove trend / make stationary."""
        if order <= 0:
            return self

        for _ in range(order):
            self.df[self.target_col] = self.df[self.target_col].diff()
        self.df = self.df.dropna()
        self.preprocessing_log.append(f"Applied differencing of order {order}.")
        return self

    # ---------- Simple smoothing ----------
    def apply_rolling_smoothing(self, window=None):
        """Optional smoothing for anomaly/clustering tasks."""
        if window and window > 1:
            self.df[self.target_col] = (
                self.df[self.target_col].rolling(window=window, center=False).mean().bfill()
            )
            self.preprocessing_log.append(f"Applied rolling mean smoothing (window={window}).")
        return self

    # ---------- Output ----------
    def get_processed_data(self):
        return self.df

    def get_log(self):
        return self.preprocessing_log
