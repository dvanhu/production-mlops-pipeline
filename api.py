from fastapi import FastAPI
import mlflow
import pandas as pd
import os

app = FastAPI()

MLFLOW_TRACKING_URI = "file:./mlruns"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

def load_model():
    try:
        runs = mlflow.search_runs(order_by=["start_time DESC"])
        if runs.empty:
            raise Exception("No MLflow runs found")

        latest_run_id = runs.iloc[0]["run_id"]
        model_uri = f"runs:/{latest_run_id}/model"

        return mlflow.sklearn.load_model(model_uri)
    except Exception as e:
        print("Model load failed:", e)
        return None

model = load_model()

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: dict):
    if model is None:
        return {"error": "Model not loaded"}

    df = pd.DataFrame([data])
    preds = model.predict(df)
    return {"prediction": preds.tolist()}
