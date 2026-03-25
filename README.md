# Invoice System (ML)

Simple training + Streamlit frontend to predict freight cost from `inventory.db` (or `data.csv` if present).

## Setup
```bash
pip install -r requirements.txt
```

## Train
```bash
python train.py
```

This saves:
- `model.pkl`
- `scaler.pkl`
- `feature_columns.pkl`

## Run Frontend (Streamlit)
```bash
streamlit run streamlit_app.py
```

Open the URL printed in your terminal. Use **Predict** to upload a CSV compatible with `vendor_invoice` (target can be `Freight`; `data.csv` is optional). 
