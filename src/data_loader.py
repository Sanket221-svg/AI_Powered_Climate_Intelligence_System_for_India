"""
Data loading utilities
"""
import os
import pandas as pd

from config import DATA_PATH


def load_data(path: str = None):
    if path is None:
        path = DATA_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")
    df = pd.read_csv(path)
    return df


def get_data_summary(df: pd.DataFrame):
    summary = {
        "rows": len(df),
        "columns": list(df.columns),
        "cities": df["City"].nunique() if "City" in df.columns else None,
        "states": df["State"].nunique() if "State" in df.columns else None,
        "date_range": None,
    }
    if "Date" in df.columns:
        try:
            df_date = pd.to_datetime(df["Date"])
            summary["date_range"] = (str(df_date.min().date()), str(df_date.max().date()))
        except Exception:
            pass
    return summary
