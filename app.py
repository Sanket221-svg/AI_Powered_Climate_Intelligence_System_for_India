"""
Streamlit app for AI-Powered Climate Intelligence System for India
"""
import os
import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, confusion_matrix, classification_report

from config import DATA_PATH, METADATA_PATH
from src.data_loader import load_data, get_data_summary
from src.preprocessing import basic_clean, feature_engineering, encode_categoricals
from predict import load_models


st.set_page_config(page_title="AI-Powered Climate Intelligence System for India", layout="wide")

st.title("AI-Powered Climate Intelligence System for India")
st.markdown("A working ML-powered dashboard for climate insights, predictions, clustering, and anomaly detection.")

# Load data
try:
    df = load_data(DATA_PATH)
    df_clean = basic_clean(df)
    df_fe = feature_engineering(df_clean)
    summary = get_data_summary(df)
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

models = load_models()

# Sidebar navigation
page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Temperature Prediction", "AQI Classification", "Climate Analytics", "Clustering and Anomalies", "Model Information"],
)

if page == "Overview":
    st.header("Overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", summary.get("rows"))
    with col2:
        st.metric("Cities", summary.get("cities"))
    with col3:
        st.metric("States", summary.get("states"))
    with col4:
        if summary.get("date_range"):
            st.metric("Date Range", f"{summary['date_range'][0]} to {summary['date_range'][1]}")
    
    st.subheader("Dataset Preview")
    st.dataframe(df_fe.head(10))
    
    st.subheader("Quick Visuals")
    col1, col2 = st.columns(2)
    with col1:
        if "Temperature_Avg (°C)" in df_fe.columns:
            fig = px.histogram(df_fe, x="Temperature_Avg (°C)", nbins=30, title="Average Temperature Distribution")
            st.plotly_chart(fig, use_container_width=True)
    with col2:
        if "AQI" in df_fe.columns and "City" in df_fe.columns:
            city_aqi = df_fe.groupby("City")["AQI"].mean().sort_values(ascending=False).head(10)
            fig = px.bar(city_aqi, x=city_aqi.index, y=city_aqi.values, title="Top 10 Cities by Avg AQI")
            st.plotly_chart(fig, use_container_width=True)

elif page == "Temperature Prediction":
    st.header("Temperature Prediction (Average Temperature)")
    st.markdown("*Result is a model estimate.*")
    
    if "temp" not in models:
        st.warning("Temperature model not found. Please run `python train_model.py` to train and save models.")
    else:
        metadata = models.get("metadata", {})
        temp_meta = metadata.get("temperature", {})
        features = temp_meta.get("features", [])
        st.write("Required features (from trained model):", features)
        
        # Build input form based on features
        input_data = {}
        col1, col2 = st.columns(2)
        # Simple form - use common features
        with col1:
            if "City" in features:
                input_data["City"] = st.selectbox("City", sorted(df_fe["City"].unique()))
            if "State" in features:
                input_data["State"] = st.selectbox("State", sorted(df_fe["State"].unique()))
            if "Humidity (%)" in features:
                input_data["Humidity (%)"] = st.number_input("Humidity (%)", value=float(df_fe["Humidity (%)"].mean()))
            if "Rainfall (mm)" in features:
                input_data["Rainfall (mm)"] = st.number_input("Rainfall (mm)", value=float(df_fe["Rainfall (mm)"].mean()))
            if "Wind_Speed (km/h)" in features:
                input_data["Wind_Speed (km/h)"] = st.number_input("Wind_Speed (km/h)", value=float(df_fe["Wind_Speed (km/h)"].mean()))
        with col2:
            if "Pressure (hPa)" in features:
                input_data["Pressure (hPa)"] = st.number_input("Pressure (hPa)", value=float(df_fe["Pressure (hPa)"].mean()))
            if "AQI" in features:
                input_data["AQI"] = st.number_input("AQI", value=float(df_fe["AQI"].mean()))
            if "Month" in features:
                input_data["Month"] = st.slider("Month", 1, 12, 6)
            if "Season" in features:
                input_data["Season"] = st.selectbox("Season", ["Winter", "Summer", "Monsoon", "Post-Monsoon"])
        
        if st.button("Predict Temperature"):
            try:
                # Prepare dataframe
                input_df = pd.DataFrame([input_data])
                # Add derived features if expected
                input_df_fe = feature_engineering(input_df)
                # Encode
                encoders = models.get("encoders", {})
                input_encoded, _ = encode_categoricals(input_df_fe, fit=False, label_encoders=encoders)
                # Ensure columns match
                for f in features:
                    if f not in input_encoded.columns:
                        input_encoded[f] = 0
                X_pred = input_encoded[features]
                pred = models["temp"].predict(X_pred)[0]
                st.success(f"Predicted Average Temperature: {pred:.2f} °C")
                if temp_meta.get("metrics"):
                    st.info(f"Model metrics (from training): MAE={temp_meta['metrics']['mae']:.3f}, RMSE={temp_meta['metrics']['rmse']:.3f}, R²={temp_meta['metrics']['r2']:.3f}")
            except Exception as e:
                st.error(f"Prediction error: {e}")

elif page == "AQI Classification":
    st.header("AQI Classification")
    st.markdown("*This predicts AQI category from features; it is not a direct AQI measurement.*")
    
    if "aqi" not in models:
        st.warning("AQI classifier not found. Please run `python train_model.py` to train and save models.")
    else:
        metadata = models.get("metadata", {})
        aqi_meta = metadata.get("aqi", {})
        features = aqi_meta.get("features", [])
        st.write("Required features (from trained model):", features)
        
        input_data = {}
        col1, col2 = st.columns(2)
        with col1:
            if "Temperature_Avg (°C)" in features:
                input_data["Temperature_Avg (°C)"] = st.number_input("Temperature_Avg (°C)", value=float(df_fe["Temperature_Avg (°C)"].mean()))
            if "Humidity (%)" in features:
                input_data["Humidity (%)"] = st.number_input("Humidity (%)", value=float(df_fe["Humidity (%)"].mean()))
            if "Rainfall (mm)" in features:
                input_data["Rainfall (mm)"] = st.number_input("Rainfall (mm)", value=float(df_fe["Rainfall (mm)"].mean()))
            if "Wind_Speed (km/h)" in features:
                input_data["Wind_Speed (km/h)"] = st.number_input("Wind_Speed (km/h)", value=float(df_fe["Wind_Speed (km/h)"].mean()))
        with col2:
            if "Pressure (hPa)" in features:
                input_data["Pressure (hPa)"] = st.number_input("Pressure (hPa)", value=float(df_fe["Pressure (hPa)"].mean()))
            if "AQI" in features:
                input_data["AQI"] = st.number_input("AQI", value=float(df_fe["AQI"].mean()))
            if "Month" in features:
                input_data["Month"] = st.slider("Month", 1, 12, 6)
        
        if st.button("Predict AQI Category"):
            try:
                input_df = pd.DataFrame([input_data])
                input_df_fe = feature_engineering(input_df)
                encoders = models.get("encoders", {})
                input_encoded, _ = encode_categoricals(input_df_fe, fit=False, label_encoders=encoders)
                for f in features:
                    if f not in input_encoded.columns:
                        input_encoded[f] = 0
                X_pred = input_encoded[features]
                pred = models["aqi"].predict(X_pred)[0]
                st.success(f"Predicted AQI Category: {pred}")
                if aqi_meta.get("metrics_test"):
                    m = aqi_meta["metrics_test"]
                    st.info(f"Test metrics: Acc={m['accuracy']:.3f}, F1={m['f1_weighted']:.3f}")
            except Exception as e:
                st.error(f"Prediction error: {e}")

elif page == "Climate Analytics":
    st.header("Climate Analytics")
    col1, col2 = st.columns(2)
    with col1:
        if "Month" in df_fe.columns and "Rainfall (mm)" in df_fe.columns:
            monthly = df_fe.groupby("Month")["Rainfall (mm)"].mean()
            fig = px.line(monthly, x=monthly.index, y=monthly.values, markers=True, title="Monthly Avg Rainfall")
            st.plotly_chart(fig, use_container_width=True)
    with col2:
        if "Season" in df_fe.columns and "Temperature_Avg (°C)" in df_fe.columns:
            season_temp = df_fe.groupby("Season")["Temperature_Avg (°C)"].mean()
            fig = px.bar(season_temp, x=season_temp.index, y=season_temp.values, title="Avg Temperature by Season")
            st.plotly_chart(fig, use_container_width=True)
    
    if "State" in df_fe.columns and "Rainfall (mm)" in df_fe.columns:
        state_rain = df_fe.groupby("State")["Rainfall (mm)"].mean().sort_values(ascending=False)
        fig = px.bar(state_rain, x=state_rain.index, y=state_rain.values, title="Avg Rainfall by State")
        st.plotly_chart(fig, use_container_width=True)

elif page == "Clustering and Anomalies":
    st.header("Clustering and Anomalies")
    if "kmeans" not in models or "scaler" not in models or "iso" not in models:
        st.warning("Clustering/Anomaly models not found. Run `python train_model.py`.")
    else:
        try:
            encoders = models.get("encoders", {})
            df_enc, _ = encode_categoricals(df_fe, fit=False, label_encoders=encoders)
            cluster_features = models["metadata"].get("clustering", {}).get("features", ["Temperature_Avg (°C)", "Humidity (%)", "Rainfall (mm)", "Wind_Speed (km/h)", "Pressure (hPa)", "AQI"])
            cluster_features = [c for c in cluster_features if c in df_enc.columns]
            scaled = models["scaler"].transform(df_enc[cluster_features])
            df_enc["Cluster"] = models["kmeans"].predict(scaled)
            df_enc["Anomaly"] = models["iso"].predict(scaled)
            st.subheader("Cluster Summary (means)")
            st.dataframe(df_enc.groupby("Cluster")[cluster_features].mean())
            fig = px.scatter(df_enc.sample(min(1000, len(df_enc))), x=cluster_features[0], y=cluster_features[1], color="Cluster", title="Cluster Visualization")
            st.plotly_chart(fig, use_container_width=True)
            st.subheader(f"Anomalies detected: {(df_enc['Anomaly']==-1).sum()}")
            st.dataframe(df_enc[df_enc["Anomaly"]==-1][["Date", "City", "State"] + cluster_features].head(20))
            st.caption("Cluster labels are arbitrary groups. Anomalies are statistical outliers.")
        except Exception as e:
            st.error(f"Error: {e}")

elif page == "Model Information":
    st.header("Model Information")
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH) as f:
            meta = json.load(f)
        st.json(meta)
    else:
        st.warning("Metadata not found. Train models first with `python train_model.py`.")
    st.subheader("Notes")
    st.markdown("- Models use preprocessing consistent with the notebook.\n- AQI classification excludes using AQI as target leakage? Check features.\n- Predictions are estimates, not forecasts unless model supports it.")
