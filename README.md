# AI-Powered Climate Intelligence System for India

End-to-end ML project with Streamlit dashboard for temperature prediction, AQI classification, clustering, and anomaly detection.

## Quick start (Windows PowerShell)

```powershell
cd "C:\Users\Sanket Chaudhari\OneDrive\Documents\Anaconda (anaconda3)\Whether Prediction system\climate-intelligence-system"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Add dataset
Place `Indian_Climate_Dataset_2024_2025.csv` in `data/` (already copied if running from this setup).

## Train models
```powershell
python train_model.py
```

## Run app
```powershell
streamlit run app.py
```

## Notes
- Do not commit large CSVs or model binaries. See `.gitignore`.
- If models are missing, the app will show instructions to train.
- This project uses models trained from the provided dataset with preprocessing consistent with the notebook.
# AI_Powered_Climate_Intelligence_System_for_India
cc3e0c297080cbd573eed825c808ebbd6b288d05
