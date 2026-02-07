import pandas as pd
import numpy as np
from datetime import datetime

def load_csv(file):
    """Load CSV file and return DataFrame"""
    try:
        df = pd.read_csv(file)
        return df, None
    except Exception as e:
        return None, str(e)

def detect_datetime_column(df):
    """Auto-detect datetime column"""
    for col in df.columns:
        try:
            pd.to_datetime(df[col], errors='raise')
            return col
        except:
            continue
    return None

def detect_frequency(df, time_col):
    """Detect time series frequency"""
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col)
    
    time_diffs = df[time_col].diff().dropna()
    median_diff = time_diffs.median()
    
    # Detect frequency based on median difference
    if median_diff <= pd.Timedelta(hours=1):
        return 'H', 'Hourly'
    elif median_diff <= pd.Timedelta(days=1):
        return 'D', 'Daily'
    elif median_diff <= pd.Timedelta(days=7):
        return 'W', 'Weekly'
    elif median_diff <= pd.Timedelta(days=31):
        return 'M', 'Monthly'
    elif median_diff <= pd.Timedelta(days=92):
        return 'Q', 'Quarterly'
    else:
        return 'Y', 'Yearly'

def validate_timeseries(df, time_col, target_col):
    """Validate time series data"""
    errors = []
    
    # Check minimum length
    if len(df) < 10:
        errors.append("Dataset too small (minimum 10 observations required)")
    
    # Check for nulls in target
    null_pct = df[target_col].isnull().sum() / len(df) * 100
    if null_pct > 50:
        errors.append(f"Too many missing values in target column ({null_pct:.1f}%)")
    
    # Check if target is numeric
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        errors.append("Target column must be numeric")
    
    # Check for duplicate timestamps
    if df[time_col].duplicated().any():
        errors.append("Duplicate timestamps detected")
    
    return len(errors) == 0, errors

def get_basic_stats(df, target_col):
    """Get basic statistics"""
    series = df[target_col].dropna()
    
    return {
        'count': len(df),
        'missing': df[target_col].isnull().sum(),
        'missing_pct': df[target_col].isnull().sum() / len(df) * 100,
        'mean': series.mean(),
        'median': series.median(),
        'std': series.std(),
        'min': series.min(),
        'max': series.max(),
        'range': series.max() - series.min()
    }

def prepare_timeseries(df, time_col, target_col):
    """Prepare time series for analysis"""
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col)
    df = df.set_index(time_col)
    
    # Handle missing values (forward fill)
    df[target_col] = df[target_col].ffill().bfill()
    
    return df
