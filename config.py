"""
Configuration for AI-Powered Climate Intelligence System for India
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

DATA_PATH = os.path.join(DATA_DIR, "Indian_Climate_Dataset_2024_2025.csv")

# Model paths
TEMP_MODEL_PATH = os.path.join(MODEL_DIR, "temperature_model.joblib")
AQI_MODEL_PATH = os.path.join(MODEL_DIR, "aqi_classifier.joblib")
CLUSTER_MODEL_PATH = os.path.join(MODEL_DIR, "kmeans_model.joblib")
ANOMALY_MODEL_PATH = os.path.join(MODEL_DIR, "isolation_forest.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "clustering_scaler.joblib")
ENCODERS_PATH = os.path.join(MODEL_DIR, "label_encoders.joblib")
METADATA_PATH = os.path.join(MODEL_DIR, "model_metadata.json")

RANDOM_STATE = 42
TEST_SIZE = 0.2
