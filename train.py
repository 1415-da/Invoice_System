import pickle
from sklearn.ensemble import RandomForestRegressor

from data_preprocessing import load_data, preprocess_data, split_and_scale
from model_evaluation import evaluate_model


def train_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


if __name__ == "__main__":
    print("Starting pipeline...\n")

    # 1. Load Data
    df = load_data("data.csv")
    print("Data loaded")

    # 2. Preprocess
    X, y = preprocess_data(df)
    print("Data preprocessed")

    # 3. Split + Scale
    X_train, X_test, y_train, y_test = split_and_scale(X, y)
    print("Data split and scaled")

    # 4. Train
    model = train_model(X_train, y_train)
    print("Model trained")

    # 5. Evaluate
    mae, mse, r2 = evaluate_model(model, X_test, y_test)

    print("\nEvaluation Results:")
    print(f"MAE: {mae}")
    print(f"MSE: {mse}")
    print(f"R2 Score: {r2}")

    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("\nModel saved as model.pkl")
    print("Pipeline completed successfully!")