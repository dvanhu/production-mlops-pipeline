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
    """Train model and log with MLflow (CI-safe version)."""

    # -----------------------------
    # Ensure MLflow works everywhere
    # -----------------------------
    os.makedirs("mlruns", exist_ok=True)

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("e2e_use_case_training")

    # -----------------------------
    # Train model
    # -----------------------------
    logger.info(f"Training model {model}...")
    model.fit(
        dataset_trn.drop(columns=[target]),
        dataset_trn[target],
    )

    # -----------------------------
    # Log model (CRITICAL)
    # -----------------------------
    with mlflow.start_run():
        mlflow.sklearn.log_model(model, "model")

    logger.info("Model logged successfully to MLflow")

    return model
