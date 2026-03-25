import pickle
from sklearn.ensemble import RandomForestRegressor

from data_preprocessing import load_data, preprocess_data, split_and_scale
from model_evaluation import evaluate_model


def train_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


def run_training(data_path: str = "data.csv"):
    # 1. Load Data
    df = load_data(data_path)

    # 2. Preprocess
    X, y = preprocess_data(df)

    # 3. Split + Scale
    X_train, X_test, y_train, y_test = split_and_scale(X, y)

    # 4. Train
    model = train_model(X_train, y_train)

    # 5. Evaluate
    mae, mse, r2 = evaluate_model(model, X_test, y_test)

    # Save artifacts for the Streamlit frontend.
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)

    return {"mae": mae, "mse": mse, "r2": r2}


if __name__ == "__main__":
    print("Starting pipeline...\n")
    metrics = run_training("data.csv")

    print("\nEvaluation Results:")
    print(f"MAE: {metrics['mae']}")
    print(f"MSE: {metrics['mse']}")
    print(f"R2 Score: {metrics['r2']}")

    print("\nModel saved as model.pkl")
    print("Pipeline completed successfully!")