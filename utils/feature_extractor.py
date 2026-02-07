import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
from scipy import stats

def extract_basic_features(df, target_col, freq):
    """Extract basic dataset features"""
    series = df[target_col].dropna()
    
    return {
        'dataset_size': len(series),
        'date_range': f"{df.index.min()} to {df.index.max()}",
        'frequency': freq,
        'missing_pct': (df[target_col].isnull().sum() / len(df)) * 100,
        'univariate': True
    }

def extract_statistical_features(series):
    """Extract statistical features"""
    # Outlier detection using IQR
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((series < (Q1 - 1.5 * IQR)) | (series > (Q3 + 1.5 * IQR))).sum()
    
    return {
        'mean': series.mean(),
        'median': series.median(),
        'std': series.std(),
        'min': series.min(),
        'max': series.max(),
        'skewness': stats.skew(series),
        'kurtosis': stats.kurtosis(series),
        'cv': series.std() / series.mean() if series.mean() != 0 else 0,
        'outliers_count': outliers,
        'outlier_pct': (outliers / len(series)) * 100
    }

def extract_trend_features(series):
    """Extract trend features"""
    # Simple linear regression on time index
    x = np.arange(len(series))
    y = series.values
    
    # Remove NaNs
    mask = ~np.isnan(y)
    x_clean = x[mask]
    y_clean = y[mask]
    
    if len(x_clean) < 2:
        return {
            'has_trend': False,
            'trend_direction': 'none',
            'trend_strength': 0,
            'trend_slope': 0,
            'trend_linearity': 0
        }
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_clean, y_clean)
    
    # Decomposition for trend strength
    try:
        if len(series) >= 24:  # Need enough data for decomposition
            decomposition = seasonal_decompose(series, model='additive', period=min(12, len(series)//2), extrapolate_trend='freq')
            trend_var = decomposition.trend.var()
            resid_var = decomposition.resid.var()
            if pd.notna(trend_var) and pd.notna(resid_var) and (trend_var + resid_var) > 0:
                trend_strength = max(0, 1 - (resid_var / (trend_var + resid_var)))
            else:
                trend_strength = abs(r_value)
        else:
            trend_strength = abs(r_value)
    except:
        trend_strength = abs(r_value)
    
    has_trend = abs(slope) > 0.01 and p_value < 0.05
    
    return {
        'has_trend': has_trend,
        'trend_direction': 'upward' if slope > 0 else 'downward' if slope < 0 else 'none',
        'trend_strength': float(trend_strength),
        'trend_slope': float(slope),
        'trend_linearity': float(r_value ** 2)
    }

def extract_seasonality_features(series, freq_code):
    """Extract seasonality features"""
    # Map frequency to seasonal period
    freq_map = {'H': 24, 'D': 7, 'W': 52, 'M': 12, 'Q': 4, 'Y': 1}
    seasonal_period = freq_map.get(freq_code, 12)
    
    if len(series) < 2 * seasonal_period:
        return {
            'has_seasonality': False,
            'seasonal_period': None,
            'seasonality_strength': 0
        }
    
    try:
        decomposition = seasonal_decompose(series, model='additive', period=seasonal_period, extrapolate_trend='freq')
        seasonal_var = decomposition.seasonal.var()
        resid_var = decomposition.resid.var()
        
        if pd.notna(seasonal_var) and pd.notna(resid_var) and (seasonal_var + resid_var) > 0:
            seasonal_strength = max(0, 1 - (resid_var / (seasonal_var + resid_var)))
        else:
            seasonal_strength = 0
        
        has_seasonality = seasonal_strength > 0.3
        
        return {
            'has_seasonality': has_seasonality,
            'seasonal_period': seasonal_period if has_seasonality else None,
            'seasonality_strength': float(seasonal_strength)
        }
    except:
        return {
            'has_seasonality': False,
            'seasonal_period': None,
            'seasonality_strength': 0
        }

def extract_stationarity_features(series):
    """Extract stationarity features"""
    try:
        adf_result = adfuller(series.dropna())
        is_stationary = adf_result[1] < 0.05
        
        return {
            'adf_statistic': float(adf_result[0]),
            'adf_p_value': float(adf_result[1]),
            'is_stationary': is_stationary
        }
    except:
        return {
            'adf_statistic': None,
            'adf_p_value': None,
            'is_stationary': False
        }

def extract_autocorrelation_features(series):
    """Extract autocorrelation features"""
    try:
        # Compute ACF and PACF
        nlags = min(40, len(series)//2)
        acf_values = acf(series.dropna(), nlags=nlags)
        pacf_values = pacf(series.dropna(), nlags=nlags)
        
        # Find significant lags (beyond 95% confidence interval)
        conf_interval = 1.96 / np.sqrt(len(series))
        
        # Significant PACF lags (AR order)
        significant_pacf = np.where(np.abs(pacf_values[1:]) > conf_interval)[0] + 1
        pacf_order = int(significant_pacf[0]) if len(significant_pacf) > 0 else 0
        
        # Recommended lags based on significant ACF
        significant_acf = np.where(np.abs(acf_values[1:]) > conf_interval)[0] + 1
        recommended_lags = significant_acf[:5].tolist() if len(significant_acf) > 0 else [1]
        
        return {
            'lag_1_acf': float(acf_values[1]) if len(acf_values) > 1 else 0,
            'pacf_order': int(pacf_order),
            'recommended_lags': [int(x) for x in recommended_lags],
            'max_acf_lag': int(np.argmax(np.abs(acf_values[1:]))) + 1
        }
    except:
        return {
            'lag_1_acf': 0,
            'pacf_order': 1,
            'recommended_lags': [1],
            'max_acf_lag': 1
        }

def compute_complexity_score(features):
    """Compute overall complexity score (0-10)"""
    score = 0
    
    # Add points for complexity factors
    if features.get('has_trend'): score += 2
    if features.get('has_seasonality'): score += 2
    if not features.get('is_stationary'): score += 1
    if features.get('outlier_pct', 0) > 5: score += 1
    if features.get('dataset_size', 0) > 1000: score += 2
    if features.get('lag_1_acf', 0) > 0.7: score += 1
    if features.get('seasonality_strength', 0) > 0.6: score += 1
    
    return min(10, score)

def compute_all_features(df, target_col, freq_code):
    """Master function to compute all features"""
    series = df[target_col]
    
    features = {}
    
    # Extract all feature groups
    features.update(extract_basic_features(df, target_col, freq_code))
    features.update(extract_statistical_features(series))
    features.update(extract_trend_features(series))
    features.update(extract_seasonality_features(series, freq_code))
    features.update(extract_stationarity_features(series))
    features.update(extract_autocorrelation_features(series))
    
    # Compute complexity score
    features['data_complexity_score'] = compute_complexity_score(features)
    
    return features
