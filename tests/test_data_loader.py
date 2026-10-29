# Tests for climate intelligence system

import os
import pandas as pd
from src.data_loader import load_data


def test_load_data():
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "Indian_Climate_Dataset_2024_2025.csv")
    if os.path.exists(path):
        df = load_data(path)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
    else:
        pass
