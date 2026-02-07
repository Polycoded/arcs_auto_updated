import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

class ModelTrainer:
    """Train and benchmark multiple forecasting models on a single series."""

    def __init__(self, df, target_col, train_size=0.8):
        self.df = df.copy()
        self.target_col = target_col
        self.train_size_ratio = train_size
        self.train_size = int(len(df) * train_size)
        self.train = df.iloc[:self.train_size]
        self.test = df.iloc[self.train_size:]
        self.results = []

    # ---------- Core metrics ----------
    def _metrics(self, y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-8, None))) * 100
        r2 = r2_score(y_true, y_pred)
        return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R²": r2}

    # ---------- Basic models ----------
    def train_sarima(self, seasonal_period=12):
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX

            model = SARIMAX(
                self.train[self.target_col],
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, seasonal_period),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            fitted = model.fit(disp=False, maxiter=50)
            preds = fitted.forecast(steps=len(self.test))

            m = self._metrics(self.test[self.target_col], preds)
            m.update(
                {
                    "name": "SARIMA",
                    "status": "success",
                    "predictions": np.asarray(preds),
                    "model_obj": fitted,
                }
            )
            return m
        except Exception as e:
            return {"name": "SARIMA", "status": "failed", "error": str(e)}

    def train_arima(self):
        try:
            from statsmodels.tsa.arima.model import ARIMA

            model = ARIMA(self.train[self.target_col], order=(1, 1, 1))
            fitted = model.fit()
            preds = fitted.forecast(steps=len(self.test))

            m = self._metrics(self.test[self.target_col], preds)
            m.update(
                {
                    "name": "ARIMA",
                    "status": "success",
                    "predictions": np.asarray(preds),
                    "model_obj": fitted,
                }
            )
            return m
        except Exception as e:
            return {"name": "ARIMA", "status": "failed", "error": str(e)}

    def train_prophet(self):
        try:
            from prophet import Prophet

            train_df = pd.DataFrame(
                {"ds": self.train.index, "y": self.train[self.target_col].values}
            )

            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                seasonality_mode="additive",
            )
            model.fit(train_df)

            future = pd.DataFrame({"ds": self.test.index})
            forecast = model.predict(future)
            preds = forecast["yhat"].values

            m = self._metrics(self.test[self.target_col], preds)
            m.update(
                {
                    "name": "Prophet",
                    "status": "success",
                    "predictions": preds,
                    "model_obj": model,
                }
            )
            return m
        except Exception as e:
            return {"name": "Prophet", "status": "failed", "error": str(e)}

    def train_ets(self, seasonal_period=12):
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing

            model = ExponentialSmoothing(
                self.train[self.target_col],
                trend="add",
                seasonal="add",
                seasonal_periods=seasonal_period,
            )
            fitted = model.fit()
            preds = fitted.forecast(steps=len(self.test))

            m = self._metrics(self.test[self.target_col], preds)
            m.update(
                {
                    "name": "Exponential Smoothing",
                    "status": "success",
                    "predictions": np.asarray(preds),
                    "model_obj": fitted,
                }
            )
            return m
        except Exception as e:
            return {"name": "Exponential Smoothing", "status": "failed", "error": str(e)}

    def train_xgboost(self, lags=None):
        try:
            from xgboost import XGBRegressor

            if lags is None:
                lags = [1, 2, 3, 7, 14]

            df_lags = self.df.copy()
            for lag in lags:
                df_lags[f"lag_{lag}"] = df_lags[self.target_col].shift(lag)

            df_lags = df_lags.dropna()
            feature_cols = [c for c in df_lags.columns if c.startswith("lag_")]
            X = df_lags[feature_cols]
            y = df_lags[self.target_col]

            train_size = int(len(df_lags) * self.train_size_ratio)
            X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
            y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

            model = XGBRegressor(
                n_estimators=200, learning_rate=0.05, max_depth=5, subsample=0.9, colsample_bytree=0.9, random_state=42
            )
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            m = self._metrics(y_test, preds)
            m.update(
                {"name": "XGBoost", "status": "success", "predictions": preds, "model_obj": model}
            )
            return m
        except Exception as e:
            return {"name": "XGBoost", "status": "failed", "error": str(e)}

    # ---------- Hybrids ----------

    def train_stl_ets(self, seasonal_period=12):
        """STL decomposition + ETS (Walmart-style hybrid)."""
        try:
            from statsmodels.tsa.seasonal import STL
            from statsmodels.tsa.holtwinters import ExponentialSmoothing

            series = self.train[self.target_col]
            stl = STL(series, period=seasonal_period, robust=True)
            res = stl.fit()

            seasonally_adjusted = res.trend + res.resid

            ets = ExponentialSmoothing(
                seasonally_adjusted, trend="add", seasonal=None
            ).fit()
            ets_forecast = ets.forecast(steps=len(self.test))

            last_season = res.seasonal.iloc[-seasonal_period:]
            seasonal_forecast = np.tile(last_season.values, int(np.ceil(len(self.test) / seasonal_period)))[
                : len(self.test)
            ]

            final_forecast = ets_forecast + seasonal_forecast

            m = self._metrics(self.test[self.target_col], final_forecast)
            m.update(
                {
                    "name": "STL + ETS (Hybrid)",
                    "status": "success",
                    "predictions": final_forecast,
                    "model_obj": (stl, ets),
                }
            )
            return m
        except Exception as e:
            return {"name": "STL + ETS (Hybrid)", "status": "failed", "error": str(e)}

    def train_arima_ets_gbdt(self, seasonal_period=12):
        """ARIMA + ETS boosted with GBDT (Rossmann-style hybrid)."""
        try:
            from statsmodels.tsa.arima.model import ARIMA
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            from xgboost import XGBRegressor

            y_train = self.train[self.target_col]

            arima = ARIMA(y_train, order=(1, 1, 1)).fit()
            ets = ExponentialSmoothing(
                y_train, trend="add", seasonal="add", seasonal_periods=seasonal_period
            ).fit()

            arima_fit = arima.fittedvalues
            ets_fit = ets.fittedvalues
            base_fit = 0.5 * (arima_fit + ets_fit)

            aligned = y_train.iloc[-len(base_fit) :]
            resid = aligned - base_fit

            df_resid = pd.DataFrame({"resid": resid})
            for lag in [1, 2, 3]:
                df_resid[f"lag_{lag}"] = df_resid["resid"].shift(lag)
            df_resid = df_resid.dropna()

            X = df_resid[[c for c in df_resid.columns if c.startswith("lag_")]]
            y = df_resid["resid"]

            gbdt = XGBRegressor(
                n_estimators=200, learning_rate=0.05, max_depth=3, subsample=0.9, colsample_bytree=0.9, random_state=42
            )
            gbdt.fit(X, y)

            arima_fc = arima.forecast(steps=len(self.test))
            ets_fc = ets.forecast(steps=len(self.test))
            base_fc = 0.5 * (arima_fc + ets_fc)

            last = resid.iloc[-3:].values.tolist()
            X_future = []
            for _ in range(len(self.test)):
                X_future.append(last[-3:])
                last.append(0.0)
            X_future = np.array(X_future)
            resid_fc = gbdt.predict(X_future)

            final_fc = base_fc.values + resid_fc

            m = self._metrics(self.test[self.target_col], final_fc)
            m.update(
                {
                    "name": "ARIMA + ETS + GBDT (Hybrid)",
                    "status": "success",
                    "predictions": final_fc,
                    "model_obj": (arima, ets, gbdt),
                }
            )
            return m
        except Exception as e:
            return {"name": "ARIMA + ETS + GBDT (Hybrid)", "status": "failed", "error": str(e)}
