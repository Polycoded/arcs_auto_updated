import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import stats

def generate_trend_evidence(series):
    evidence = []
    x = np.arange(len(series))
    y = series.values
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    if p_value < 0.05:
        direction = "upward" if slope > 0 else "downward"
        evidence.append(f"Linear regression: significant {direction} trend (p = {p_value:.4f}).")
        evidence.append(f"Slope = {slope:.4f}, R² = {r_value**2:.3f}.")
    else:
        evidence.append(f"Linear regression: no significant linear trend (p = {p_value:.4f}).")
    return evidence

def generate_seasonality_evidence(series, period):
    evidence = []
    try:
        decomp = seasonal_decompose(series, model="additive", period=period, extrapolate_trend="freq")
        s_var = decomp.seasonal.var()
        r_var = decomp.resid.var()
        evidence.append(f"Seasonal variance = {s_var:.3f}, residual variance = {r_var:.3f}.")
        if s_var + r_var > 0:
            strength = max(0, 1 - r_var / (s_var + r_var))
            evidence.append(f"Seasonality strength ≈ {strength*100:.1f}%.")
            if strength > 0.6:
                evidence.append("Strong repeating seasonal pattern present.")
            elif strength > 0.3:
                evidence.append("Moderate seasonal pattern present.")
            else:
                evidence.append("Only weak seasonal pattern detected.")
    except Exception as e:
        evidence.append(f"Could not compute decomposition: {e}")
    return evidence

def generate_stationarity_evidence(series):
    evidence = []
    try:
        stat, p, lags, nobs, crit, _ = adfuller(series.dropna())
        evidence.append(f"ADF test: statistic = {stat:.3f}, p = {p:.4f}.")
        if p < 0.05:
            evidence.append("ADF suggests the series is stationary (rejects unit root).")
        else:
            evidence.append("ADF suggests the series is non-stationary (fails to reject unit root).")
    except Exception as e:
        evidence.append(f"ADF test failed: {e}")

    try:
        stat, p, lags, crit = kpss(series.dropna(), regression="c", nlags="auto")
        evidence.append(f"KPSS test: statistic = {stat:.3f}, p = {p:.4f}.")
        if p > 0.05:
            evidence.append("KPSS suggests the series is stationary.")
        else:
            evidence.append("KPSS suggests the series is non-stationary.")
    except Exception as e:
        evidence.append(f"KPSS test failed: {e}")
    return evidence

def generate_autocorrelation_evidence(series):
    evidence = []
    nlags = min(40, len(series)//2)
    acf_vals = acf(series.dropna(), nlags=nlags)
    pacf_vals = pacf(series.dropna(), nlags=nlags)

    ci = 1.96 / np.sqrt(len(series))
    sig_acf = np.where(np.abs(acf_vals[1:]) > ci)[0] + 1
    sig_pacf = np.where(np.abs(pacf_vals[1:]) > ci)[0] + 1

    if len(sig_acf):
        evidence.append(f"ACF: significant correlation at lags {sig_acf[:5].tolist()}.")
    else:
        evidence.append("ACF: no clearly significant lags.")
    if len(sig_pacf):
        evidence.append(f"PACF: suggests AR order around {int(sig_pacf[0])}.")
    return evidence

def summarize_patterns(features):
    """Plain‑English summary of trend, seasonality, cyclicity, stationarity."""
    parts = []

    # Trend
    if features.get("has_trend"):
        direction = features.get("trend_direction", "none")
        strength = features.get("trend_strength", 0)
        if strength > 0.7:
            parts.append(f"The data has a strong {direction} trend over time.")
        elif strength > 0.4:
            parts.append(f"The data shows a moderate {direction} trend.")
        else:
            parts.append(f"There is only a weak {direction} trend.")
    else:
        parts.append("The data does not show a clear long‑term upward or downward trend.")

    # Seasonality vs cycles
    if features.get("has_seasonality"):
        period = features.get("seasonal_period")
        msg = f"There is a clear seasonal pattern repeating every {period} time steps"
        if period == 7:
            msg += " (roughly weekly seasonality)."
        elif period == 12:
            msg += " (roughly yearly seasonality for monthly data)."
        else:
            msg += "."
        parts.append(msg)
    else:
        if features.get("data_complexity_score", 0) >= 7:
            parts.append(
                "The series does not follow a fixed calendar pattern but moves in irregular waves, "
                "which is more like **cyclical** behaviour than standard seasonality."
            )
        else:
            parts.append("No strong calendar‑based seasonal pattern is detected.")

    # Stationarity
    if features.get("is_stationary"):
        parts.append("Overall, the series looks statistically stable over time (stationary).")
    else:
        parts.append("The series is not stationary; applying differencing or detrending will help many models.")

    return parts
