"""
Preprocessing utilities for the project
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def basic_clean(df: pd.DataFrame):
    df = df.copy()
    # Convert Date
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    # Drop duplicates
    df = df.drop_duplicates()
    # Forward fill
    df = df.ffill()
    return df


def feature_engineering(df: pd.DataFrame):
    df = df.copy()
    if "Date" in df.columns and pd.api.types.is_datetime64_any_dtype(df["Date"]):
        df["Month"] = df["Date"].dt.month
        df["Year"] = df["Date"].dt.year
        df["Day"] = df["Date"].dt.day

        def season(month):
            if month in [12, 1, 2]:
                return "Winter"
            elif month in [3, 4, 5]:
                return "Summer"
            elif month in [6, 7, 8, 9]:
                return "Monsoon"
            else:
                return "Post-Monsoon"

        df["Season"] = df["Month"].apply(season)

    # Derived features as in notebook
    if "Temperature_Max (°C)" in df.columns and "Temperature_Min (°C)" in df.columns:
        df["Temp_Diff"] = df["Temperature_Max (°C)"] - df["Temperature_Min (°C)"]

    if "Rainfall (mm)" in df.columns:
        df["Rain"] = np.where(df["Rainfall (mm)"] > 0, 1, 0)

    if "Wind_Speed (km/h)" in df.columns:
        df["Wind_Category"] = pd.cut(
            df["Wind_Speed (km/h)"],
            bins=[0, 10, 20, 40],
            labels=["Low", "Medium", "High"],
        )

    return df


def encode_categoricals(df: pd.DataFrame, fit=True, label_encoders=None):
    df = df.copy()
    if label_encoders is None:
        label_encoders = {}
    for col in ["City", "State", "Season", "AQI_Category", "Wind_Category"]:
        if col not in df.columns:
            continue
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
        else:
            if col in label_encoders:
                le = label_encoders[col]
                # handle unseen
                df[col] = df[col].astype(str).apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )
            else:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                label_encoders[col] = le
    return df, label_encoders
