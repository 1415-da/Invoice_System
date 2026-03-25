import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import sqlite3
from pathlib import Path


def load_data(path):
    csv_path = Path(path)
    if csv_path.exists():
        return pd.read_csv(csv_path)

    db_path = Path("inventory.db")
    if db_path.exists():
        with sqlite3.connect(db_path) as conn:
            return pd.read_sql_query("SELECT * FROM vendor_invoice", conn)

    raise FileNotFoundError(
        f"Neither '{path}' nor 'inventory.db' were found in the project directory."
    )


def _infer_target_column(df: pd.DataFrame) -> str:
    """Return the target column name for either CSV or sqlite data."""
    if "freight_cost" in df.columns:
        return "freight_cost"
    if "Freight" in df.columns:
        return "Freight"
    raise ValueError("Target column not found. Expected 'freight_cost' or 'Freight'.")


def preprocess_data(df):
    target = _infer_target_column(df)

    df = df.dropna()
    if df.empty:
        raise ValueError("No rows left after dropna().")

    X = df.drop(columns=[target])
    y = df[target]

    # Convert non-numeric features for sklearn models.
    X = pd.get_dummies(X, drop_first=True)

    return X, y


def split_and_scale(X, y):
    # Save the exact one-hot encoded feature order used during training.
    with open("feature_columns.pkl", "wb") as f:
        pickle.dump(X.columns.tolist(), f)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Save scaler
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    return X_train, X_test, y_train, y_test


def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build feature matrix X from a dataframe for prediction.
    If the target column is present, it is dropped.
    """
    df = df.dropna()
    if df.empty:
        raise ValueError("No rows left after dropna().")

    target = None
    try:
        target = _infer_target_column(df)
    except ValueError:
        target = None

    X = df.drop(columns=[target]) if target is not None else df.copy()
    X = pd.get_dummies(X, drop_first=True)
    return X


if __name__ == "__main__":
    df = load_data("data.csv")
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_and_scale(X, y)

    print("Preprocessing Done")