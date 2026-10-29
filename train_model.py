"""
Training script for models
"""
import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, IsolationForest
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import (
    DATA_PATH,
    TEMP_MODEL_PATH,
    AQI_MODEL_PATH,
    CLUSTER_MODEL_PATH,
    ANOMALY_MODEL_PATH,
    SCALER_PATH,
    ENCODERS_PATH,
    METADATA_PATH,
    MODEL_DIR,
    RANDOM_STATE,
    TEST_SIZE,
)
from src.data_loader import load_data
from src.preprocessing import basic_clean, feature_engineering, encode_categoricals
from src.evaluation import regression_metrics, classification_metrics


def train():
    os.makedirs(MODEL_DIR, exist_ok=True)
    df = load_data(DATA_PATH)
    df = basic_clean(df)
    df = feature_engineering(df)
    df_encoded, label_encoders = encode_categoricals(df, fit=True)
    joblib.dump(label_encoders, ENCODERS_PATH)

    metadata = {"random_state": RANDOM_STATE, "test_size": TEST_SIZE}

    # Temperature prediction
    # As in notebook, drop Temperature_Avg? Wait - predict avg temp; features from others
    # Notebook did: X = df.drop(columns=["Temperature_Avg (°C)", "Date"])
    X_temp = df_encoded.drop(columns=["Temperature_Avg (°C)", "Date"], errors="ignore")
    y_temp = df_encoded["Temperature_Avg (°C)"]
    X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(X_temp, y_temp, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    rf_reg = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)
    rf_reg.fit(X_train_t, y_train_t)
    y_pred_t = rf_reg.predict(X_test_t)
    reg_metrics = regression_metrics(y_test_t, y_pred_t)
    joblib.dump(rf_reg, TEMP_MODEL_PATH)
    metadata["temperature"] = {
        "features": list(X_temp.columns),
        "metrics": reg_metrics,
    }

    # AQI classification
    # Notebook: X = df.drop(columns=["AQI_Category", "Date"]), y = df["AQI_Category"]
    X_aqi = df_encoded.drop(columns=["AQI_Category", "Date"], errors="ignore")
    y_aqi = df_encoded["AQI_Category"]
    X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(X_aqi, y_aqi, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    rf_clf.fit(X_train_a, y_train_a)
    y_pred_a = rf_clf.predict(X_test_a)
    clf_metrics = classification_metrics(y_train_a, rf_clf.predict(X_train_a))  # train not needed, but keep simple
    clf_metrics_test = classification_metrics(y_test_a, y_pred_a)
    joblib.dump(rf_clf, AQI_MODEL_PATH)
    metadata["aqi"] = {
        "features": list(X_aqi.columns),
        "metrics_test": clf_metrics_test,
    }

    # Clustering
    cluster_features = ["Temperature_Avg (°C)", "Humidity (%)", "Rainfall (mm)", "Wind_Speed (km/h)", "Pressure (hPa)", "AQI"]
    cluster_features = [c for c in cluster_features if c in df_encoded.columns]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df_encoded[cluster_features])
    joblib.dump(scaler, SCALER_PATH)
    kmeans = KMeans(n_clusters=4, random_state=RANDOM_STATE)
    kmeans.fit(scaled)
    joblib.dump(kmeans, CLUSTER_MODEL_PATH)
    metadata["clustering"] = {"features": cluster_features, "n_clusters": 4}

    # Anomaly
    iso = IsolationForest(contamination=0.02, random_state=RANDOM_STATE)
    iso.fit(scaled)
    joblib.dump(iso, ANOMALY_MODEL_PATH)
    metadata["anomaly"] = {"features": cluster_features, "contamination": 0.02}

    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    print("Training complete. Models saved.")


if __name__ == "__main__":
    train()
