"""
Flask API for AI-Powered Climate Intelligence System for India
"""
import os
import json
from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np

from config import DATA_PATH, METADATA_PATH
from src.data_loader import load_data, get_data_summary
from src.preprocessing import basic_clean, feature_engineering, encode_categoricals
from predict import load_models

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

df_raw = load_data(DATA_PATH)
df_clean = basic_clean(df_raw)
df_fe = feature_engineering(df_clean)
summary = get_data_summary(df_raw)
models = load_models()


@app.route("/")
def index():
    return render_template("index.html", summary=summary)


@app.route("/api/summary")
def api_summary():
    return jsonify(summary)


@app.route("/api/data/preview")
def api_data_preview():
    try:
        preview = df_fe.head(100).fillna(0).to_dict(orient="records")
        return jsonify(preview)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/charts/overview")
def api_charts_overview():
    try:
        data = {}
        if "Temperature_Avg (°C)" in df_fe.columns:
            data["temp_hist"] = df_fe["Temperature_Avg (°C)"].dropna().tolist()
        if "City" in df_fe.columns and "AQI" in df_fe.columns:
            city_aqi = df_fe.groupby("City")["AQI"].mean().sort_values(ascending=False).head(15)
            data["city_aqi_labels"] = city_aqi.index.tolist()
            data["city_aqi_values"] = city_aqi.values.tolist()
        if "State" in df_fe.columns and "Rainfall (mm)" in df_fe.columns:
            state_rain = df_fe.groupby("State")["Rainfall (mm)"].mean().sort_values(ascending=False).head(15)
            data["state_rain_labels"] = state_rain.index.tolist()
            data["state_rain_values"] = state_rain.values.tolist()
        if "Month" in df_fe.columns and "Rainfall (mm)" in df_fe.columns:
            monthly = df_fe.groupby("Month")["Rainfall (mm)"].mean()
            data["month_rain_labels"] = [int(x) for x in monthly.index.tolist()]
            data["month_rain_values"] = [float(x) for x in monthly.values.tolist()]
        if "Season" in df_fe.columns and "Temperature_Avg (°C)" in df_fe.columns:
            season_temp = df_fe.groupby("Season")["Temperature_Avg (°C)"].mean()
            data["season_temp_labels"] = season_temp.index.tolist()
            data["season_temp_values"] = [float(x) for x in season_temp.values.tolist()]
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/predict/temperature", methods=["POST"])
def api_predict_temperature():
    try:
        if "temp" not in models:
            return jsonify({"error": "Temperature model not found. Train models first."}), 400
        payload = request.json or {}
        metadata = models.get("metadata", {})
        features = metadata.get("temperature", {}).get("features", [])
        input_df = pd.DataFrame([payload])
        input_df_fe = feature_engineering(input_df)
        encoders = models.get("encoders", {})
        input_encoded, _ = encode_categoricals(input_df_fe, fit=False, label_encoders=encoders)
        for f in features:
            if f not in input_encoded.columns:
                input_encoded[f] = 0
        X_pred = input_encoded[features]
        pred = float(models["temp"].predict(X_pred)[0])
        result = {"prediction": pred, "unit": "°C", "note": "Model estimate"}
        if metadata.get("temperature", {}).get("metrics"):
            result["metrics"] = metadata["temperature"]["metrics"]
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/predict/aqi", methods=["POST"])
def api_predict_aqi():
    try:
        if "aqi" not in models:
            return jsonify({"error": "AQI classifier not found. Train models first."}), 400
        payload = request.json or {}
        metadata = models.get("metadata", {})
        features = metadata.get("aqi", {}).get("features", [])
        input_df = pd.DataFrame([payload])
        input_df_fe = feature_engineering(input_df)
        encoders = models.get("encoders", {})
        input_encoded, _ = encode_categoricals(input_df_fe, fit=False, label_encoders=encoders)
        for f in features:
            if f not in input_encoded.columns:
                input_encoded[f] = 0
        X_pred = input_encoded[features]
        pred = models["aqi"].predict(X_pred)[0]
        result = {"prediction": str(pred), "note": "Predicted category (not actual AQI reading)"}
        if metadata.get("aqi", {}).get("metrics_test"):
            result["metrics"] = metadata["aqi"]["metrics_test"]
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/clustering")
def api_clustering():
    try:
        if "kmeans" not in models or "scaler" not in models:
            return jsonify({"error": "Clustering models not found"}), 400
        encoders = models.get("encoders", {})
        df_enc, _ = encode_categoricals(df_fe, fit=False, label_encoders=encoders)
        cluster_features = models["metadata"].get("clustering", {}).get("features", ["Temperature_Avg (°C)", "Humidity (%)", "Rainfall (mm)", "Wind_Speed (km/h)", "Pressure (hPa)", "AQI"])
        cluster_features = [c for c in cluster_features if c in df_enc.columns]
        scaled = models["scaler"].transform(df_enc[cluster_features])
        df_enc["Cluster"] = models["kmeans"].predict(scaled)
        summary = df_enc.groupby("Cluster")[cluster_features].mean().round(2).to_dict()
        sample = df_enc.sample(min(500, len(df_enc)))[cluster_features + ["Cluster"]].fillna(0).to_dict(orient="records")
        return jsonify({"cluster_summary": summary, "sample": sample, "features": cluster_features})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/anomalies")
def api_anomalies():
    try:
        if "iso" not in models or "scaler" not in models:
            return jsonify({"error": "Anomaly model not found"}), 400
        encoders = models.get("encoders", {})
        df_enc, _ = encode_categoricals(df_fe, fit=False, label_encoders=encoders)
        cluster_features = models["metadata"].get("anomaly", {}).get("features", ["Temperature_Avg (°C)", "Humidity (%)", "Rainfall (mm)", "Wind_Speed (km/h)", "Pressure (hPa)", "AQI"])
        cluster_features = [c for c in cluster_features if c in df_enc.columns]
        scaled = models["scaler"].transform(df_enc[cluster_features])
        preds = models["iso"].predict(scaled)
        df_enc["Anomaly"] = preds
        anomalies = df_enc[df_enc["Anomaly"] == -1].head(50).fillna(0).to_dict(orient="records")
        return jsonify({"count": int((preds == -1).sum()), "anomalies": anomalies})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/model/info")
def api_model_info():
    try:
        if os.path.exists(METADATA_PATH):
            with open(METADATA_PATH) as f:
                meta = json.load(f)
            return jsonify(meta)
        else:
            return jsonify({"error": "Metadata not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
