from typing_extensions import Annotated

import os
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.base import ClassifierMixin
from zenml import ArtifactConfig, step
from zenml.logger import get_logger

logger = get_logger(__name__)


@step
def model_trainer(
    dataset_trn: pd.DataFrame,
    model: ClassifierMixin,
    target: str,
    name: str,
) -> Annotated[ClassifierMixin, ArtifactConfig(name="model", is_model_artifact=True)]:
    """Train model and log with MLflow (CI-safe, production-ready)."""

    # -----------------------------
    # MLflow Setup (robust)
    # -----------------------------
    os.makedirs("mlruns", exist_ok=True)

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("e2e_use_case_training")

    # Enable autologging (VERY IMPORTANT)
    mlflow.sklearn.autolog()

    # -----------------------------
    # Train + Log
    # -----------------------------
    logger.info(f"Training model {model}...")

    with mlflow.start_run(run_name=f"{name}_training_run"):
        model.fit(
            dataset_trn.drop(columns=[target]),
            dataset_trn[target],
        )

        # Explicit model logging (extra safety)
        mlflow.sklearn.log_model(model, "model")

    logger.info("Model trained and logged successfully to MLflow")

    return model
