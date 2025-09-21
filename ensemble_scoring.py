import numpy as np
import pandas as pd
import joblib

# Dummy weights example – יש לכוונן עבור כל אופק בהתאם למשקלת המודלים
WEIGHTS = {
    30: {"xgb_cls": 0.25, "xgb_reg": 0.15, "tech": 0.20, "sent": 0.10, "fund": 0.10, "lstm": 0.10, "mc": 0.10},
    60: {"xgb_cls": 0.23, "xgb_reg": 0.15, "tech": 0.16, "sent": 0.08, "fund": 0.13, "lstm": 0.12, "mc": 0.13},
    90: {"xgb_cls": 0.22, "xgb_reg": 0.15, "tech": 0.15, "sent": 0.08, "fund": 0.14, "lstm": 0.12, "mc": 0.14},
    120: {"xgb_cls": 0.20, "xgb_reg": 0.15, "tech": 0.14, "sent": 0.08, "fund": 0.15, "lstm": 0.12, "mc": 0.16},
    180: {"xgb_cls": 0.18, "xgb_reg": 0.15, "tech": 0.13, "sent": 0.07, "fund": 0.20, "lstm": 0.10, "mc": 0.17},
}

# Model component descriptions for UI
MODEL_DESCRIPTIONS = {
    "xgb_cls": "XGBoost Classifier",
    "xgb_reg": "XGBoost Regressor", 
    "tech": "Technical Analysis",
    "sent": "Sentiment Analysis",
    "fund": "Fundamental Analysis",
    "lstm": "LSTM Neural Network",
    "mc": "Monte Carlo Simulation"
}

def scale_score(x):
    """Scale score to [-1, 1] range"""
    return max(-1, min(1, x))

def combine(horizon, components: dict):
    """
    Combine model components using weighted ensemble for given horizon
    
    Args:
        horizon (int): Prediction horizon in days (30, 60, 90, 120, 180)
        components (dict): Dictionary of model component scores
        
    Returns:
        float: Combined ensemble score scaled to [-1, 1]
    """
    wmap = WEIGHTS.get(horizon)
    if wmap is None:
        raise ValueError(f"Unsupported horizon: {horizon}. Supported horizons: {list(WEIGHTS.keys())}")
    
    raw = 0
    for k, v in wmap.items():
        raw += v * components.get(k, 0)
    return scale_score(raw)

def get_available_horizons():
    """Get list of available prediction horizons"""
    return list(WEIGHTS.keys())

def get_model_weights(horizon):
    """Get model weights for a specific horizon"""
    return WEIGHTS.get(horizon, {})

def validate_components(components: dict):
    """
    Validate that component scores are in expected range [-1, 1]
    
    Args:
        components (dict): Dictionary of model component scores
        
    Returns:
        dict: Dictionary with validation results
    """
    validation_results = {}
    for component, score in components.items():
        if not isinstance(score, (int, float)):
            validation_results[component] = f"Invalid type: {type(score).__name__}"
        elif score < -1 or score > 1:
            validation_results[component] = f"Score {score} out of range [-1, 1]"
        else:
            validation_results[component] = "Valid"
    
    return validation_results

if __name__ == "__main__":
    # דוגמה עבור אופק של 60 יום
    comps = {"xgb_cls": 0.6, "xgb_reg": 0.4, "tech": 0.5, "sent": 0.2, "fund": 0.7, "lstm": 0.3, "mc": 0.45}
    print("Score 60d:", combine(60, comps))