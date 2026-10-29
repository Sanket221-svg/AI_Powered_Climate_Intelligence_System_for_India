"""
Prediction helpers
"""
import os
import joblib
import json
import pandas as pd

from config import (
    TEMP_MODEL_PATH,
    AQI_MODEL_PATH,
    CLUSTER_MODEL_PATH,
    ANOMALY_MODEL_PATH,
    SCALER_PATH,
    ENCODERS_PATH,
    METADATA_PATH,
)


def load_models():
    models = {}
    if os.path.exists(TEMP_MODEL_PATH):
        models["temp"] = joblib.load(TEMP_MODEL_PATH)
    if os.path.exists(AQI_MODEL_PATH):
        models["aqi"] = joblib.load(AQI_MODEL_PATH)
    if os.path.exists(CLUSTER_MODEL_PATH):
        models["kmeans"] = joblib.load(CLUSTER_MODEL_PATH)
    if os.path.exists(ANOMALY_MODEL_PATH):
        models["iso"] = joblib.load(ANOMALY_MODEL_PATH)
    if os.path.exists(SCALER_PATH):
        models["scaler"] = joblib.load(SCALER_PATH)
    if os.path.exists(ENCODERS_PATH):
        models["encoders"] = joblib.load(ENCODERS_PATH)
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH) as f:
            models["metadata"] = json.load(f)
    return models


def predict_temperature(features_df: pd.DataFrame, models):
    if "temp" not in models:
        raise ValueError("Temperature model not loaded")
    model = models["temp"]
    return model.predict(features_df)
