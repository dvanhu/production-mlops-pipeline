import os
import mlflow
import pandas as pd
from sklearn.base import ClassifierMixin
from zenml import step
from zenml.logger import get_logger

logger = get_logger(__name__)


@step
def model_evaluator(
    model: ClassifierMixin,
    dataset_trn: pd.DataFrame,
    dataset_tst: pd.DataFrame,
    target: str,
    min_train_accuracy: float = 0.0,
    min_test_accuracy: float = 0.0,
    fail_on_accuracy_quality_gates: bool = False,
) -> None:
    """Evaluate model and log metrics (CI-safe MLflow version)."""

    # -----------------------------
    # MLflow Setup (MUST MATCH TRAINER)
    # -----------------------------
    os.makedirs("mlruns", exist_ok=True)

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("e2e_use_case_training")

    # -----------------------------
    # Accuracy Calculation
    # -----------------------------
    trn_acc = model.score(
        dataset_trn.drop(columns=[target]),
        dataset_trn[target],
    )
    logger.info(f"Train accuracy={trn_acc*100:.2f}%")

    tst_acc = model.score(
        dataset_tst.drop(columns=[target]),
        dataset_tst[target],
    )
    logger.info(f"Test accuracy={tst_acc*100:.2f}%")

    # -----------------------------
    # Log Metrics (same run context style)
    # -----------------------------
    with mlflow.start_run(run_name="model_evaluation_run"):
        mlflow.log_metric("training_accuracy_score", trn_acc)
        mlflow.log_metric("testing_accuracy_score", tst_acc)

    # -----------------------------
    # Quality Gates
    # -----------------------------
    messages = []

    if trn_acc < min_train_accuracy:
        messages.append(
            f"Train accuracy {trn_acc*100:.2f}% is below {min_train_accuracy*100:.2f}%!"
        )

    if tst_acc < min_test_accuracy:
        messages.append(
            f"Test accuracy {tst_acc*100:.2f}% is below {min_test_accuracy*100:.2f}%!"
        )

    if fail_on_accuracy_quality_gates and messages:
        raise RuntimeError(
            "Model performance did not meet the minimum criteria:\n"
            + "\n".join(messages)
        )
    else:
        for message in messages:
            logger.warning(message)
