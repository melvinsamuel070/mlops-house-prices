import os
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import mlflow
import mlflow.sklearn
from datetime import datetime
import numpy as np

# =====================================
# Configuration
# =====================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")
RESULTS_DIR = os.path.join(BASE_DIR, "..", "results")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Detect if running in GitHub Actions
IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"

# MLflow Tracking URI Setup
if IN_GITHUB_ACTIONS:
    # ✅ Ensure mlruns is inside the repo and writable
    MLFLOW_DIR = os.path.join(BASE_DIR, "..", "mlruns")
    os.makedirs(MLFLOW_DIR, exist_ok=True)
    MLFLOW_TRACKING_URI = f"file:{MLFLOW_DIR}"
else:
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# =====================================
# Helper functions
# =====================================

def load_data():
    housing = fetch_california_housing(as_frame=True)
    return housing.data, housing.target, housing.feature_names

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    return {
        'mse': mse,
        'mae': mean_absolute_error(y_test, preds),
        'r2': r2_score(y_test, preds),
        'rmse': rmse
    }

def save_metrics_to_file(metrics, filepath):
    with open(filepath, 'w') as f:
        f.write("Model Evaluation Metrics\n")
        f.write("=" * 30 + "\n")
        for name, value in metrics.items():
            f.write(f"{name.upper()}: {value:.6f}\n")
        f.write(f"\nGenerated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

def save_predictions_to_file(y_test, preds, filepath):
    df = pd.DataFrame({
        'actual': y_test,
        'predicted': preds,
        'error': y_test - preds
    })
    df.to_csv(filepath, index=False)

# =====================================
# Main function
# =====================================

def main():
    X, y, feature_names = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    mlflow.set_experiment("california_house_price_prediction")
    run_name = f"linear_regression_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("features", str(feature_names))

        model = LinearRegression()
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        metrics = evaluate_model(model, X_test, y_test)

        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        # ✅ Modern MLflow logging (fixed `name` deprecation)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=X_train.iloc[:5],
            registered_model_name="CaliforniaHousePriceModel"
        )

        for feature, coef in zip(feature_names, model.coef_):
            mlflow.log_metric(f"coef_{feature}", coef)
        mlflow.log_metric("intercept", model.intercept_)

        # Save and log result files
        metrics_file = os.path.join(RESULTS_DIR, "model_metrics.txt")
        predictions_file = os.path.join(RESULTS_DIR, "predictions.csv")
        model_summary_file = os.path.join(RESULTS_DIR, "model_summary.txt")

        save_metrics_to_file(metrics, metrics_file)
        save_predictions_to_file(y_test, preds, predictions_file)

        with open(model_summary_file, 'w') as f:
            f.write("Linear Regression Model Summary\n")
            f.write("=" * 40 + "\n")
            for feature, coef in zip(feature_names, model.coef_):
                f.write(f"{feature}: {coef:.6f}\n")
            f.write(f"Intercept: {model.intercept_:.6f}\n")
            f.write(f"\nGenerated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        mlflow.log_artifact(metrics_file, "results")
        mlflow.log_artifact(predictions_file, "results")
        mlflow.log_artifact(model_summary_file, "results")

        # Save model locally
        model_path = os.path.join(MODEL_DIR, "house_price_model.pkl")
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path, "local_model")

        print("✅ Model training complete!")
        print(f"📍 MLflow Tracking URI: {MLFLOW_TRACKING_URI}")
        print(f"📊 Metrics logged to MLflow at {MLFLOW_TRACKING_URI}")
        print(f"💾 Model saved locally at: {model_path}")


if __name__ == "__main__":
    main()
