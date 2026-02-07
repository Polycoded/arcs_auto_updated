import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf

def plot_timeseries(df, target_col, title="Time Series"):
    """Plot time series with plotly"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df[target_col],
        mode='lines',
        name=target_col,
        line=dict(color='#1f77b4', width=2)
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Value",
        hovermode='x unified',
        template='plotly_white',
        height=400
    )
    
    return fig

def plot_decomposition(series, freq_code):
    """Plot time series decomposition"""
    freq_map = {'H': 24, 'D': 7, 'W': 52, 'M': 12, 'Q': 4, 'Y': 1}
    period = freq_map.get(freq_code, 12)
    
    if len(series) < 2 * period:
        return None
    
    try:
        decomposition = seasonal_decompose(series, model='additive', period=period, extrapolate_trend='freq')
        
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=('Observed', 'Trend', 'Seasonal', 'Residual'),
            vertical_spacing=0.08
        )
        
        # Observed
        fig.add_trace(go.Scatter(x=series.index, y=series, mode='lines', name='Observed',
                                line=dict(color='#1f77b4')), row=1, col=1)
        
        # Trend
        fig.add_trace(go.Scatter(x=decomposition.trend.index, y=decomposition.trend,
                                mode='lines', name='Trend', line=dict(color='#ff7f0e')), row=2, col=1)
        
        # Seasonal
        fig.add_trace(go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal,
                                mode='lines', name='Seasonal', line=dict(color='#2ca02c')), row=3, col=1)
        
        # Residual
        fig.add_trace(go.Scatter(x=decomposition.resid.index, y=decomposition.resid,
                                mode='lines', name='Residual', line=dict(color='#d62728')), row=4, col=1)
        
        fig.update_layout(height=800, showlegend=False, template='plotly_white',
                         title_text="Time Series Decomposition")
        fig.update_xaxes(title_text="Time", row=4, col=1)
        
        return fig
    except Exception as e:
        return None

def plot_acf_pacf(series):
    """Plot ACF and PACF"""
    try:
        nlags = min(40, len(series)//2)
        acf_values = acf(series.dropna(), nlags=nlags)
        pacf_values = pacf(series.dropna(), nlags=nlags)
        
        conf_interval = 1.96 / np.sqrt(len(series))
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Autocorrelation Function (ACF)', 'Partial Autocorrelation Function (PACF)')
        )
        
        # ACF
        fig.add_trace(go.Bar(x=list(range(len(acf_values))), y=acf_values, name='ACF',
                            marker_color='#1f77b4'), row=1, col=1)
        fig.add_hline(y=conf_interval, line_dash="dash", line_color="red", row=1, col=1)
        fig.add_hline(y=-conf_interval, line_dash="dash", line_color="red", row=1, col=1)
        
        # PACF
        fig.add_trace(go.Bar(x=list(range(len(pacf_values))), y=pacf_values, name='PACF',
                            marker_color='#ff7f0e'), row=1, col=2)
        fig.add_hline(y=conf_interval, line_dash="dash", line_color="red", row=1, col=2)
        fig.add_hline(y=-conf_interval, line_dash="dash", line_color="red", row=1, col=2)
        
        fig.update_layout(height=400, showlegend=False, template='plotly_white')
        fig.update_xaxes(title_text="Lag", row=1, col=1)
        fig.update_xaxes(title_text="Lag", row=1, col=2)
        fig.update_yaxes(title_text="Correlation", row=1, col=1)
        fig.update_yaxes(title_text="Correlation", row=1, col=2)
        
        return fig
    except:
        return None

def create_stats_cards(features):
    """Format features for card display"""
    cards = []
    
    # Basic Info
    cards.append({
        'title': '📊 Dataset Info',
        'items': [
            f"Size: {features['dataset_size']} observations",
            f"Frequency: {features['frequency']}",
            f"Missing: {features['missing_pct']:.1f}%",
            f"Outliers: {features['outlier_pct']:.1f}%"
        ]
    })
    
    # Trend
    cards.append({
        'title': '📈 Trend Analysis',
        'items': [
            f"Trend: {'Yes' if features['has_trend'] else 'No'}",
            f"Direction: {features['trend_direction'].title()}",
            f"Strength: {features['trend_strength']:.2f}",
            f"Slope: {features['trend_slope']:.4f}"
        ]
    })
    
    # Seasonality
    cards.append({
        'title': '🔄 Seasonality',
        'items': [
            f"Seasonal: {'Yes' if features['has_seasonality'] else 'No'}",
            f"Period: {features['seasonal_period'] if features['seasonal_period'] else 'N/A'}",
            f"Strength: {features['seasonality_strength']:.2f}",
            ""
        ]
    })
    
    # Stationarity
    cards.append({
        'title': '⚖️ Stationarity',
        'items': [
            f"Stationary: {'Yes' if features['is_stationary'] else 'No'}",
            f"ADF p-value: {features['adf_p_value']:.4f}" if features['adf_p_value'] else "ADF: N/A",
            f"Interpretation: {'Stationary' if features['is_stationary'] else 'Needs differencing'}",
            ""
        ]
    })
    
    # Autocorrelation
    cards.append({
        'title': '🔗 Autocorrelation',
        'items': [
            f"Lag-1 ACF: {features['lag_1_acf']:.3f}",
            f"AR order hint: {features['pacf_order']}",
            f"Key lags: {features['recommended_lags'][:3]}",
            f"Strength: {'Strong' if abs(features['lag_1_acf']) > 0.7 else 'Moderate' if abs(features['lag_1_acf']) > 0.3 else 'Weak'}"
        ]
    })
    
    # Complexity
    cards.append({
        'title': '🎯 Complexity Score',
        'items': [
            f"Score: {features['data_complexity_score']}/10",
            f"Level: {'Simple' if features['data_complexity_score'] < 4 else 'Moderate' if features['data_complexity_score'] < 7 else 'Complex'}",
            f"Hybrid: {'Recommended' if features['data_complexity_score'] >= 7 else 'Optional'}",
            ""
        ]
    })
    
    return cards

def plot_trend_simple(df, target_col, window=7):
    """Simple trend plot: raw series + rolling mean."""
    series = df[target_col]
    roll = series.rolling(window=window, center=False).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=series.index, y=series,
        mode="lines", name="Original",
        line=dict(color="#d3d3d3", width=1),
        opacity=0.7,
    ))
    fig.add_trace(go.Scatter(
        x=roll.index, y=roll,
        mode="lines", name=f"{window}-point average (trend)",
        line=dict(color="#1f77b4", width=3),
    ))
    fig.update_layout(
        title=f"Simple Trend View ({window}-point moving average)",
        xaxis_title="Time",
        yaxis_title="Value",
        template="plotly_white",
        height=400,
    )
    return fig


def plot_seasonality_simple(df, time_col_name, target_col, freq_code):
    """
    Simple seasonal plot:
    - If daily data -> average by day of week
    - If monthly data -> average by month
    """
    df_plot = df.copy()
    df_plot["__time"] = df_plot.index  # index is already datetime

    if freq_code in ["D", "H"]:  # daily / hourly -> show day-of-week pattern
        df_plot["dow"] = df_plot["__time"].dt.dayofweek
        avg = df_plot.groupby("dow")[target_col].mean().reindex(range(7))
        labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        title = "Average value by day of week (seasonality)"
        x = labels
        y = avg.values
    elif freq_code in ["M", "Q", "Y"]:
        df_plot["month"] = df_plot["__time"].dt.month
        avg = df_plot.groupby("month")[target_col].mean().reindex(range(1, 13))
        labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        title = "Average value by month (seasonality)"
        x = labels
        y = avg.values
    else:
        # fallback: group by month
        df_plot["month"] = df_plot["__time"].dt.month
        avg = df_plot.groupby("month")[target_col].mean().reindex(range(1, 13))
        labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        title = "Average value by month (seasonality)"
        x = labels
        y = avg.values

    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=y, marker_color="#2ca02c"))
    fig.update_layout(
        title=title,
        xaxis_title="",
        yaxis_title="Average value",
        template="plotly_white",
        height=400,
    )
    return fig

