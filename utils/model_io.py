import os
import joblib

MODELS_DIR = "saved_models"
os.makedirs(MODELS_DIR, exist_ok=True)

def save_model(model_obj, name: str):
    """Save model object to disk."""
    safe_name = name.replace(" ", "_").lower()
    path = os.path.join(MODELS_DIR, f"{safe_name}.joblib")
    joblib.dump(model_obj, path)
    return path

def load_model(name: str):
    """Load model object from disk."""
    safe_name = name.replace(" ", "_").lower()
    path = os.path.join(MODELS_DIR, f"{safe_name}.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No saved model at {path}")
    return joblib.load(path)
