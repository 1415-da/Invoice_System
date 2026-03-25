import pickle
from io import BytesIO

import pandas as pd
import streamlit as st

from data_preprocessing import preprocess_features
from train import run_training


MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"
FEATURES_PATH = "feature_columns.pkl"


def _load_artifacts():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(FEATURES_PATH, "rb") as f:
        feature_columns = pickle.load(f)
    return model, scaler, feature_columns


def _align_features(X: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    # Add missing one-hot columns as zeros, and drop extras.
    aligned = X.copy()
    for col in feature_columns:
        if col not in aligned.columns:
            aligned[col] = 0
    aligned = aligned[feature_columns]
    return aligned


st.set_page_config(page_title="Invoice Freight Predictor", layout="wide")
st.title("Invoice Freight Predictor")

with st.sidebar:
    st.header("Model")
    train_clicked = st.button("Train / Retrain model", type="primary")

    st.divider()
    st.caption("Training uses `inventory.db` if `data.csv` is missing.")

if train_clicked:
    with st.spinner("Training model..."):
        metrics = run_training("data.csv")
    st.success("Model trained successfully.")
    st.metric("MAE", metrics["mae"])
    st.metric("MSE", metrics["mse"])
    st.metric("R2", metrics["r2"])


st.subheader("Predict")
uploaded = st.file_uploader(
    "Upload a CSV with columns from `vendor_invoice` (Freight or freight_cost is optional).",
    type=["csv"],
)

if uploaded is None:
    st.info("Upload a CSV to generate predictions.")
    st.stop()

try:
    model, scaler, feature_columns = _load_artifacts()
except FileNotFoundError:
    st.warning("Missing `model.pkl`, `scaler.pkl`, or `feature_columns.pkl`. Click Train first.")
    st.stop()

bytes_data = uploaded.read()
df = pd.read_csv(BytesIO(bytes_data))

df_clean = df.dropna().copy()
X = preprocess_features(df_clean)
X = _align_features(X, feature_columns)

# Scale + predict
X_scaled = scaler.transform(X)
y_pred = model.predict(X_scaled)

out = df_clean.copy()
out["predicted_freight"] = y_pred
st.dataframe(out)

