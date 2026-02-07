def recommend_for_forecasting(features):
    """Rule-based recommendations for forecasting"""
    recommendations = []
    
    size = features['dataset_size']
    has_trend = features['has_trend']
    has_seasonality = features['has_seasonality']
    is_stationary = features['is_stationary']
    complexity = features['data_complexity_score']
    seasonal_period = features.get('seasonal_period')
    seasonality_strength = features['seasonality_strength']
    
    # Rule 1: SARIMA - Strong seasonality + non-stationary
    if has_seasonality and not is_stationary and size < 5000:
        confidence = 90 + (seasonality_strength * 10)
        recommendations.append({
            'name': 'SARIMA',
            'confidence': min(95, confidence),
            'category': 'Statistical',
            'reasons': [
                f"Strong seasonality detected (period={seasonal_period})",
                "Non-stationary data (needs differencing)",
                "Medium dataset size suitable for SARIMA",
                f"Seasonal strength: {seasonality_strength:.2f}"
            ],
            'pros': [
                "Handles both trend and seasonality",
                "Well-tested classical method",
                "Interpretable parameters",
                "Works well with seasonal patterns"
            ],
            'cons': [
                "Requires careful parameter tuning",
                "Assumes linear relationships",
                "Can be slow on very large datasets"
            ]
        })
    
    # Rule 2: Prophet - Multiple seasonalities or business data
    if has_seasonality and size >= 100:
        confidence = 80 + (seasonality_strength * 15)
        recommendations.append({
            'name': 'Prophet',
            'confidence': min(92, confidence),
            'category': 'ML',
            'reasons': [
                "Handles multiple seasonal patterns well",
                "Robust to missing data and outliers",
                "Good for business time series",
                "Minimal parameter tuning required"
            ],
            'pros': [
                "Easy to use, minimal tuning",
                "Handles holidays and special events",
                "Works with irregular data",
                "Provides uncertainty intervals"
            ],
            'cons': [
                "Less control than SARIMA",
                "Can overfit with too many parameters",
                "Requires sufficient historical data"
            ]
        })
    
    # Rule 3: ARIMA - Trend but no seasonality
    if has_trend and not has_seasonality and size < 3000:
        confidence = 85
        recommendations.append({
            'name': 'ARIMA',
            'confidence': confidence,
            'category': 'Statistical',
            'reasons': [
                "Trend detected without seasonality",
                f"Non-stationary: {not is_stationary}",
                "Classical approach suitable for this pattern",
                f"Dataset size: {size} observations"
            ],
            'pros': [
                "Well-established methodology",
                "Good for univariate forecasting",
                "Handles non-stationarity via differencing",
                "Interpretable results"
            ],
            'cons': [
                "Requires stationarity transformation",
                "Parameter selection can be complex",
                "Assumes linear relationships"
            ]
        })
    
    # Rule 4: Exponential Smoothing - Simple patterns, small data
    if complexity < 5 and size < 1000:
        confidence = 75
        recommendations.append({
            'name': 'Exponential Smoothing (ETS)',
            'confidence': confidence,
            'category': 'Statistical',
            'reasons': [
                "Simple pattern detected",
                "Small to medium dataset",
                "Fast and efficient method",
                f"Complexity score: {complexity}/10"
            ],
            'pros': [
                "Very fast computation",
                "Simple to understand and implement",
                "Good for short-term forecasts",
                "Minimal computational resources"
            ],
            'cons': [
                "Limited to simple patterns",
                "No external variables support",
                "May underperform on complex data"
            ]
        })
    
    # Rule 5: XGBoost with lags - Large dataset, complex patterns
    if size > 1000 or complexity >= 6:
        confidence = 82 + (min(complexity, 10) * 1.5)
        recommendations.append({
            'name': 'XGBoost with Lag Features',
            'confidence': min(92, confidence),
            'category': 'ML',
            'reasons': [
                f"Large dataset ({size} observations)",
                f"Complex patterns (score: {complexity}/10)",
                "Can capture non-linear relationships",
                f"Recommended lags: {features['recommended_lags'][:3]}"
            ],
            'pros': [
                "Handles non-linear patterns well",
                "Can include external features",
                "Fast training and prediction",
                "Robust to outliers"
            ],
            'cons': [
                "Requires feature engineering",
                "Less interpretable than statistical models",
                "Needs careful lag selection"
            ]
        })
    
    # Rule 6: LSTM - Very large dataset, complex patterns
    if size > 5000 and complexity >= 7:
        confidence = 78 + (complexity * 2)
        recommendations.append({
            'name': 'LSTM (Deep Learning)',
            'confidence': min(88, confidence),
            'category': 'Deep Learning',
            'reasons': [
                f"Very large dataset ({size} observations)",
                "Highly complex patterns detected",
                "Can capture long-term dependencies",
                f"Strong autocorrelation: {features['lag_1_acf']:.2f}"
            ],
            'pros': [
                "Captures complex sequential patterns",
                "Handles long-term dependencies",
                "Can model multivariate inputs",
                "State-of-the-art for complex data"
            ],
            'cons': [
                "Requires large amounts of data",
                "Computationally expensive",
                "Difficult to interpret",
                "Needs careful hyperparameter tuning"
            ]
        })
    
    # Sort by confidence
    recommendations.sort(key=lambda x: x['confidence'], reverse=True)
    
    # Prepare response
    result = {
        'primary': recommendations[0] if recommendations else None,
        'alternatives': recommendations[1:3] if len(recommendations) > 1 else []
    }
    
    # Hybrid suggestion for complex data
    if complexity >= 7:
        result['hybrid_suggestion'] = {
            'combo': 'SARIMA + XGBoost Ensemble',
            'rationale': f'High complexity score ({complexity}/10) suggests combining statistical and ML approaches for better accuracy',
            'approach': 'Use SARIMA for trend/seasonality, XGBoost for residuals and non-linear patterns'
        }
    
    return result

def recommend_for_classification(features):
    """Rule-based recommendations for classification"""
    recommendations = []
    size = features['dataset_size']
    
    recommendations.append({
        'name': 'Random Forest Classifier',
        'confidence': 85,
        'category': 'ML',
        'reasons': [
            "Robust ensemble method",
            "Handles non-linear patterns well",
            "Works with extracted time series features"
        ],
        'pros': ["Easy to use", "Handles outliers", "Feature importance"],
        'cons': ["Can overfit", "Less interpretable"]
    })
    
    if size > 1000:
        recommendations.append({
            'name': 'LSTM Classifier',
            'confidence': 80,
            'category': 'Deep Learning',
            'reasons': [
                "Large dataset available",
                "Captures sequential patterns",
                "End-to-end learning"
            ],
            'pros': ["Learns from raw sequences", "High accuracy potential"],
            'cons': ["Requires more data", "Complex to tune"]
        })
    
    recommendations.append({
        'name': 'k-NN with DTW',
        'confidence': 75,
        'category': 'ML',
        'reasons': [
            "Distance-based classification",
            "Works well with time series similarity",
            "Interpretable results"
        ],
        'pros': ["Simple method", "No training required", "Intuitive"],
        'cons': ["Slow on large datasets", "Sensitive to noise"]
    })
    
    return {
        'primary': recommendations[0],
        'alternatives': recommendations[1:],
        'hybrid_suggestion': None
    }

def recommend_for_anomaly_detection(features):
    """Rule-based recommendations for anomaly detection"""
    recommendations = []
    size = features['dataset_size']
    has_seasonality = features['has_seasonality']
    
    if has_seasonality:
        recommendations.append({
            'name': 'STL Decomposition + Z-Score',
            'confidence': 88,
            'category': 'Statistical',
            'reasons': [
                "Strong seasonality detected",
                "Decompose first, then detect anomalies in residuals",
                "Interpretable and effective"
            ],
            'pros': ["Handles seasonal patterns", "Easy to interpret", "Fast"],
            'cons': ["Needs clear seasonality", "Fixed threshold"]
        })
    
    recommendations.append({
        'name': 'Isolation Forest',
        'confidence': 85,
        'category': 'ML',
        'reasons': [
            "General-purpose anomaly detection",
            "No assumptions about distribution",
            "Efficient on large datasets"
        ],
        'pros': ["Fast and scalable", "No distribution assumptions", "Robust"],
        'cons': ["Less interpretable", "Requires tuning contamination"]
    })
    
    if size > 500:
        recommendations.append({
            'name': 'LSTM Autoencoder',
            'confidence': 82,
            'category': 'Deep Learning',
            'reasons': [
                "Sufficient data for deep learning",
                "Learns normal patterns automatically",
                "Captures complex sequences"
            ],
            'pros': ["Learns complex patterns", "No manual feature engineering"],
            'cons': ["Needs more data", "Computationally expensive"]
        })
    
    return {
        'primary': recommendations[0],
        'alternatives': recommendations[1:],
        'hybrid_suggestion': None
    }

def recommend_for_clustering(features):
    """Rule-based recommendations for clustering"""
    recommendations = []
    
    recommendations.append({
        'name': 'K-Means with DTW Distance',
        'confidence': 82,
        'category': 'ML',
        'reasons': [
            "Time series-aware distance metric",
            "Groups similar temporal patterns",
            "Well-established method"
        ],
        'pros': ["Works with time series shapes", "Interpretable clusters"],
        'cons': ["Need to specify k", "Sensitive to scaling"]
    })
    
    recommendations.append({
        'name': 'Hierarchical Clustering',
        'confidence': 78,
        'category': 'ML',
        'reasons': [
            "No need to specify number of clusters upfront",
            "Creates dendrogram for visualization",
            "Flexible distance metrics"
        ],
        'pros': ["Visual hierarchy", "No k required", "Interpretable"],
        'cons': ["Slower on large data", "Hard to update"]
    })
    
    recommendations.append({
        'name': 'DBSCAN',
        'confidence': 75,
        'category': 'ML',
        'reasons': [
            "Density-based clustering",
            "Finds arbitrary shaped clusters",
            "Handles noise/outliers"
        ],
        'pros': ["Finds outliers", "No k needed", "Flexible shapes"],
        'cons': ["Parameter sensitive", "Struggles with varying density"]
    })
    
    return {
        'primary': recommendations[0],
        'alternatives': recommendations[1:],
        'hybrid_suggestion': None
    }

def get_recommendations(features, task):
    """Master function to get recommendations based on task"""
    if task == 'Forecasting':
        return recommend_for_forecasting(features)
    elif task == 'Classification':
        return recommend_for_classification(features)
    elif task == 'Anomaly Detection':
        return recommend_for_anomaly_detection(features)
    elif task == 'Clustering':
        return recommend_for_clustering(features)
    else:
        return {'primary': None, 'alternatives': [], 'hybrid_suggestion': None}
