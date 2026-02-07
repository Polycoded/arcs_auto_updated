def generate_code_snippet(algorithm, features, task, time_col='date', target_col='value'):
    """Generate code snippet for selected algorithm"""
    
    templates = {
        'SARIMA': generate_sarima_code,
        'Prophet': generate_prophet_code,
        'ARIMA': generate_arima_code,
        'Exponential Smoothing (ETS)': generate_ets_code,
        'XGBoost with Lag Features': generate_xgboost_code,
        'LSTM (Deep Learning)': generate_lstm_code,
        'Random Forest Classifier': generate_rf_classifier_code,
        'LSTM Classifier': generate_lstm_classifier_code,
        'k-NN with DTW': generate_knn_dtw_code,
        'STL Decomposition + Z-Score': generate_stl_anomaly_code,
        'Isolation Forest': generate_isolation_forest_code,
        'LSTM Autoencoder': generate_lstm_autoencoder_code,
        'K-Means with DTW Distance': generate_kmeans_dtw_code,
        'Hierarchical Clustering': generate_hierarchical_code,
        'DBSCAN': generate_dbscan_code
    }
    
    generator = templates.get(algorithm)
    if generator:
        return generator(features, time_col, target_col)
    else:
        return "# Code generation not available for this algorithm yet"

def generate_sarima_code(features, time_col, target_col):
    """Generate SARIMA code"""
    seasonal_period = features.get('seasonal_period', 12)
    p = min(features.get('pacf_order', 1), 3)
    d = 0 if features['is_stationary'] else 1
    q = 1
    
    code = f"""# SARIMA Model
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Load your data
df = pd.read_csv('your_data.csv')
df['{time_col}'] = pd.to_datetime(df['{time_col}'])
df = df.set_index('{time_col}')

# Your data characteristics:
# - Seasonal period: {seasonal_period}
# - Trend: {features['trend_direction']}
# - Stationarity: {'Yes' if features['is_stationary'] else 'No (needs differencing)'}

# Train-test split (80-20)
train_size = int(len(df) * 0.8)
train = df.iloc[:train_size]
test = df.iloc[train_size:]

# Fit SARIMA model
# Order: (p, d, q) - AR, Differencing, MA
# Seasonal Order: (P, D, Q, s) - Seasonal AR, Seasonal Diff, Seasonal MA, Period
model = SARIMAX(
    train['{target_col}'],
    order=({p}, {d}, {q}),
    seasonal_order=(1, 1, 1, {seasonal_period}),
    enforce_stationarity=False,
    enforce_invertibility=False
)

results = model.fit(disp=False)
print(results.summary())

# Forecast
forecast = results.forecast(steps=len(test))

# Evaluate
mae = mean_absolute_error(test['{target_col}'], forecast)
rmse = np.sqrt(mean_squared_error(test['{target_col}'], forecast))
print(f'MAE: {{mae:.2f}}')
print(f'RMSE: {{rmse:.2f}}')

# Future forecast
future_forecast = results.forecast(steps=30)  # Next 30 periods
"""
    return code

def generate_prophet_code(features, time_col, target_col):
    """Generate Prophet code"""
    code = f"""# Prophet Model
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# Load your data
df = pd.read_csv('your_data.csv')
df['{time_col}'] = pd.to_datetime(df['{time_col}'])

# Prophet requires columns named 'ds' and 'y'
df_prophet = df.rename(columns={{'{time_col}': 'ds', '{target_col}': 'y'}})

# Train-test split
train_size = int(len(df_prophet) * 0.8)
train = df_prophet.iloc[:train_size]
test = df_prophet.iloc[train_size:]

# Initialize and fit Prophet
model = Prophet(
    yearly_seasonality={'Yes' if features['has_seasonality'] else False},
    weekly_seasonality={'Yes' if features.get('seasonal_period') == 7 else False},
    daily_seasonality=False,
    seasonality_mode='additive'  # or 'multiplicative'
)

model.fit(train)

# Make predictions
future = model.make_future_dataframe(periods=len(test))
forecast = model.predict(future)

# Evaluate on test set
test_forecast = forecast.iloc[train_size:]
mae = mean_absolute_error(test['y'], test_forecast['yhat'])
print(f'MAE: {{mae:.2f}}')

# Plot results
fig = model.plot(forecast)
plt.show()

# Plot components
fig2 = model.plot_components(forecast)
plt.show()

# Future forecast
future_30 = model.make_future_dataframe(periods=30)
future_forecast = model.predict(future_30)
"""
    return code

def generate_arima_code(features, time_col, target_col):
    """Generate ARIMA code"""
    p = min(features.get('pacf_order', 1), 3)
    d = 0 if features['is_stationary'] else 1
    
    code = f"""# ARIMA Model
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error
import numpy as np

# Load your data
df = pd.read_csv('your_data.csv')
df['{time_col}'] = pd.to_datetime(df['{time_col}'])
df = df.set_index('{time_col}')

# Train-test split
train_size = int(len(df) * 0.8)
train = df.iloc[:train_size]
test = df.iloc[train_size:]

# Fit ARIMA model
# Order: (p, d, q) = (AR order, Differencing, MA order)
model = ARIMA(train['{target_col}'], order=({p}, {d}, 1))
results = model.fit()
print(results.summary())

# Forecast
forecast = results.forecast(steps=len(test))

# Evaluate
mae = mean_absolute_error(test['{target_col}'], forecast)
print(f'MAE: {{mae:.2f}}')

# Future forecast
future_forecast = results.forecast(steps=30)
"""
    return code

def generate_ets_code(features, time_col, target_col):
    """Generate Exponential Smoothing code"""
    code = f"""# Exponential Smoothing Model
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error

# Load your data
df = pd.read_csv('your_data.csv')
df['{time_col}'] = pd.to_datetime(df['{time_col}'])
df = df.set_index('{time_col}')

# Train-test split
train_size = int(len(df) * 0.8)
train = df.iloc[:train_size]
test = df.iloc[train_size:]

# Fit Exponential Smoothing
model = ExponentialSmoothing(
    train['{target_col}'],
    trend='{"add" if features["has_trend"] else None}',
    seasonal='{"add" if features["has_seasonality"] else None}',
    seasonal_periods={features.get('seasonal_period', 12) if features['has_seasonality'] else None}
)
results = model.fit()

# Forecast
forecast = results.forecast(steps=len(test))

# Evaluate
mae = mean_absolute_error(test['{target_col}'], forecast)
print(f'MAE: {{mae:.2f}}')
"""
    return code

def generate_xgboost_code(features, time_col, target_col):
    """Generate XGBoost code"""
    lags = features.get('recommended_lags', [1, 2, 3])[:5]
    
    code = f"""# XGBoost with Lag Features
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

# Load your data
df = pd.read_csv('your_data.csv')
df['{time_col}'] = pd.to_datetime(df['{time_col}'])
df = df.sort_values('{time_col}')

# Create lag features
lags = {lags}
for lag in lags:
    df[f'lag_{{lag}}'] = df['{target_col}'].shift(lag)

# Create time features
df['day_of_week'] = df['{time_col}'].dt.dayofweek
df['month'] = df['{time_col}'].dt.month
df['day_of_month'] = df['{time_col}'].dt.day

# Drop rows with NaN (from lagging)
df = df.dropna()

# Prepare features and target
feature_cols = [f'lag_{{lag}}' for lag in lags] + ['day_of_week', 'month', 'day_of_month']
X = df[feature_cols]
y = df['{target_col}']

# Train-test split
train_size = int(len(df) * 0.8)
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

# Train XGBoost
model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, y_pred)
print(f'MAE: {{mae:.2f}}')

# Feature importance
import matplotlib.pyplot as plt
from xgboost import plot_importance
plot_importance(model)
plt.show()
"""
    return code

def generate_lstm_code(features, time_col, target_col):
    """Generate LSTM code"""
    code = f"""# LSTM Model
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_absolute_error

# Load your data
df = pd.read_csv('your_data.csv')
df['{time_col}'] = pd.to_datetime(df['{time_col}'])
df = df.sort_values('{time_col}')

# Normalize data
scaler = MinMaxScaler()
data = scaler.fit_transform(df[['{target_col}']].values)

# Create sequences
def create_sequences(data, seq_length=30):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 30
X, y = create_sequences(data, seq_length)

# Train-test split
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build LSTM model
model = Sequential([
    LSTM(50, activation='relu', return_sequences=True, input_shape=(seq_length, 1)),
    Dropout(0.2),
    LSTM(50, activation='relu'),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# Train
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# Predict
y_pred = model.predict(X_test)

# Inverse transform
y_test_inv = scaler.inverse_transform(y_test)
y_pred_inv = scaler.inverse_transform(y_pred)

# Evaluate
mae = mean_absolute_error(y_test_inv, y_pred_inv)
print(f'MAE: {{mae:.2f}}')
"""
    return code

def generate_rf_classifier_code(features, time_col, target_col):
    """Generate Random Forest Classifier code"""
    code = f"""# Random Forest Classifier for Time Series
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute

# Load your data (should have 'id', 'time', 'value', 'label' columns)
df = pd.read_csv('your_data.csv')

# Extract time series features using tsfresh
features = extract_features(df, column_id='id', column_sort='time')
features = impute(features)

# Get labels
labels = df.groupby('id')['label'].first()

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=42
)

# Train Random Forest
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)

# Evaluate
print(f'Accuracy: {{accuracy_score(y_test, y_pred):.2f}}')
print(classification_report(y_test, y_pred))
"""
    return code

def generate_lstm_classifier_code(features, time_col, target_col):
    """Generate LSTM Classifier code"""
    code = """# LSTM Classifier
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import classification_report

# Assume X_train, X_test, y_train, y_test are prepared sequences
# Shape: (samples, timesteps, features)

n_classes = len(np.unique(y_train))

model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.3),
    LSTM(32),
    Dropout(0.3),
    Dense(n_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1)

# Predict
y_pred = model.predict(X_test).argmax(axis=1)
print(classification_report(y_test, y_pred))
"""
    return code

def generate_knn_dtw_code(features, time_col, target_col):
    """Generate k-NN with DTW code"""
    code = """# k-NN with DTW Distance
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from sklearn.metrics import classification_report

# Prepare your time series data
# X_train, X_test: shape (n_samples, n_timesteps, n_features)
# y_train, y_test: labels

# Normalize
scaler = TimeSeriesScalerMeanVariance()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train k-NN with DTW
clf = KNeighborsTimeSeriesClassifier(n_neighbors=5, metric='dtw')
clf.fit(X_train_scaled, y_train)

# Predict
y_pred = clf.predict(X_test_scaled)
print(classification_report(y_test, y_pred))
"""
    return code

def generate_stl_anomaly_code(features, time_col, target_col):
    """Generate STL + anomaly detection code"""
    seasonal_period = features.get('seasonal_period', 12)
    code = f"""# STL Decomposition + Anomaly Detection
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL

# Load data
df = pd.read_csv('your_data.csv')
df['{time_col}'] = pd.to_datetime(df['{time_col}'])
df = df.set_index('{time_col}')

# STL Decomposition
stl = STL(df['{target_col}'], seasonal={seasonal_period})
result = stl.fit()

# Get residuals
residuals = result.resid

# Z-score anomaly detection
threshold = 3  # 3 standard deviations
mean = residuals.mean()
std = residuals.std()
z_scores = np.abs((residuals - mean) / std)

# Identify anomalies
anomalies = df[z_scores > threshold]
print(f'Found {{len(anomalies)}} anomalies')
print(anomalies)

# Plot
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['{target_col}'], label='Original')
plt.scatter(anomalies.index, anomalies['{target_col}'], color='red', label='Anomalies', s=50)
plt.legend()
plt.show()
"""
    return code

def generate_isolation_forest_code(features, time_col, target_col):
    """Generate Isolation Forest code"""
    code = f"""# Isolation Forest for Anomaly Detection
import pandas as pd
from sklearn.ensemble import IsolationForest

# Load data
df = pd.read_csv('your_data.csv')

# Create features (lags, rolling stats, etc.)
df['lag_1'] = df['{target_col}'].shift(1)
df['rolling_mean'] = df['{target_col}'].rolling(window=7).mean()
df['rolling_std'] = df['{target_col}'].rolling(window=7).std()
df = df.dropna()

# Prepare features
X = df[['lag_1', 'rolling_mean', 'rolling_std']]

# Train Isolation Forest
clf = IsolationForest(contamination=0.05, random_state=42)  # 5% contamination
predictions = clf.fit_predict(X)

# -1 for anomalies, 1 for normal
df['anomaly'] = predictions
anomalies = df[df['anomaly'] == -1]

print(f'Found {{len(anomalies)}} anomalies')
print(anomalies[['{target_col}']])
"""
    return code

def generate_lstm_autoencoder_code(features, time_col, target_col):
    """Generate LSTM Autoencoder code"""
    code = """# LSTM Autoencoder for Anomaly Detection
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense

# Prepare sequences (normal data only for training)
# X_train shape: (samples, timesteps, features)

timesteps = 30
n_features = 1

# Build autoencoder
input_layer = Input(shape=(timesteps, n_features))
encoded = LSTM(32, activation='relu')(input_layer)
decoded = RepeatVector(timesteps)(encoded)
decoded = LSTM(32, activation='relu', return_sequences=True)(decoded)
decoded = TimeDistributed(Dense(n_features))(decoded)

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# Train on normal data
autoencoder.fit(X_train, X_train, epochs=50, batch_size=32, validation_split=0.1)

# Detect anomalies using reconstruction error
X_test_pred = autoencoder.predict(X_test)
mse = np.mean(np.power(X_test - X_test_pred, 2), axis=1)

# Threshold: mean + 2*std
threshold = mse.mean() + 2*mse.std()
anomalies = mse > threshold
print(f'Found {anomalies.sum()} anomalies')
"""
    return code

def generate_kmeans_dtw_code(features, time_col, target_col):
    """Generate K-Means with DTW code"""
    code = """# K-Means Clustering with DTW
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
import matplotlib.pyplot as plt

# Prepare time series data
# X shape: (n_samples, n_timesteps, n_features)

# Normalize
scaler = TimeSeriesScalerMeanVariance()
X_scaled = scaler.fit_transform(X)

# K-Means with DTW
n_clusters = 3  # Adjust based on your needs
model = TimeSeriesKMeans(n_clusters=n_clusters, metric='dtw', random_state=42)
labels = model.fit_predict(X_scaled)

# Visualize clusters
plt.figure(figsize=(12, 8))
for i in range(n_clusters):
    plt.subplot(n_clusters, 1, i+1)
    for ts in X_scaled[labels == i]:
        plt.plot(ts.ravel(), alpha=0.3)
    plt.plot(model.cluster_centers_[i].ravel(), 'r-', linewidth=2)
    plt.title(f'Cluster {i+1}')
plt.tight_layout()
plt.show()
"""
    return code

def generate_hierarchical_code(features, time_col, target_col):
    """Generate Hierarchical Clustering code"""
    code = """# Hierarchical Clustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt

# Prepare your feature matrix X

# Compute linkage
Z = linkage(X, method='ward')

# Plot dendrogram
plt.figure(figsize=(12, 6))
dendrogram(Z)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()

# Fit model
n_clusters = 3  # Choose based on dendrogram
model = AgglomerativeClustering(n_clusters=n_clusters)
labels = model.fit_predict(X)

print(f'Cluster distribution: {np.bincount(labels)}')
"""
    return code

def generate_dbscan_code(features, time_col, target_col):
    """Generate DBSCAN code"""
    code = """# DBSCAN Clustering
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import numpy as np

# Prepare features
# X: your feature matrix

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# DBSCAN
model = DBSCAN(eps=0.5, min_samples=5)
labels = model.fit_predict(X_scaled)

# -1 indicates noise/outliers
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)

print(f'Number of clusters: {n_clusters}')
print(f'Number of noise points: {n_noise}')
print(f'Cluster distribution: {np.bincount(labels[labels >= 0])}')
"""
    return code
