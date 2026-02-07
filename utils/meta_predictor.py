"""
Meta-Model Predictor
Loads trained meta-model and predicts best algorithm for new time series.
"""

import json
import joblib
import numpy as np

class MetaPredictor:
    """Use trained meta-model to predict best algorithm."""
    
    def __init__(self, 
                 model_path='models/meta_model.joblib', 
                 config_path='models/meta_model_config.json'):
        try:
            # Load model
            self.model = joblib.load(model_path)
            
            # Load config
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            
            self.top_features = self.config['top_features']
            
            # Load class names (algorithm names)
            classes_file = self.config.get('classes_file', 'models/meta_model_classes.npy')
            self.classes_ = np.load(classes_file, allow_pickle=True)
            
            self.enabled = True
            print(f"✅ Meta-model loaded successfully from models/")
            print(f"   Features: {len(self.top_features)}")
            print(f"   Classes: {len(self.classes_)}")
            print(f"   Accuracy: {self.config.get('accuracy', 0)*100:.1f}%")
            
        except FileNotFoundError as e:
            print(f"⚠️ Meta-model files not found in models/ directory")
            print("   Run 'python train_meta_model.py' to train the model first.")
            self.enabled = False
        except Exception as e:
            print(f"⚠️ Error loading meta-model: {e}")
            self.enabled = False
    
    def predict(self, features_dict):
        """
        Predict best algorithm given features dictionary.
        
        Args:
            features_dict: Dictionary with feature names as keys
            
        Returns:
            (predicted_algorithm_name, confidence_score)
        """
        if not self.enabled:
            return None, 0.0
        
        try:
            # Extract only top features in correct order
            X = []
            for feat in self.top_features:
                value = features_dict.get(feat, 0)
                # Handle NaN/inf
                if np.isnan(value) or np.isinf(value):
                    value = 0
                X.append(value)
            
            X = np.array(X).reshape(1, -1)
            
            # Predict (returns encoded label)
            pred_label = self.model.predict(X)[0]
            
            # Get probabilities
            proba = self.model.predict_proba(X)[0]
            confidence = float(np.max(proba))
            
            # Decode label back to algorithm name
            pred_algo = self.classes_[int(pred_label)]
            
            return pred_algo, confidence
            
        except Exception as e:
            print(f"⚠️ Prediction error: {e}")
            return None, 0.0
    
    def predict_top_k(self, features_dict, k=3):
        """
        Predict top K algorithms with their confidence scores.
        
        Returns:
            List of (algorithm_name, confidence) tuples, sorted by confidence
        """
        if not self.enabled:
            return []
        
        try:
            # Extract features
            X = []
            for feat in self.top_features:
                value = features_dict.get(feat, 0)
                if np.isnan(value) or np.isinf(value):
                    value = 0
                X.append(value)
            
            X = np.array(X).reshape(1, -1)
            
            # Get probabilities for all classes
            proba = self.model.predict_proba(X)[0]
            
            # Get top k
            top_indices = np.argsort(proba)[::-1][:k]
            
            results = []
            for idx in top_indices:
                algo_name = self.classes_[idx]
                confidence = float(proba[idx])
                results.append((algo_name, confidence))
            
            return results
            
        except Exception as e:
            print(f"⚠️ Top-k prediction error: {e}")
            return []
