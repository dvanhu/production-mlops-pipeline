import os
import click
from datetime import datetime as dt
from typing import Optional

from zenml.logger import get_logger

from pipelines import (
    e2e_use_case_batch_inference,
    e2e_use_case_training,
    e2e_use_case_deployment,
)

logger = get_logger(__name__)

# Detect CI environment
IS_CI = os.getenv("CI", "false").lower() == "true"


@click.command()
@click.option("--no-cache", is_flag=True, default=False)
@click.option("--no-drop-na", is_flag=True, default=False)
@click.option("--no-normalize", is_flag=True, default=False)
@click.option("--drop-columns", default=None, type=str)
@click.option("--test-size", default=0.2, type=float)
@click.option("--min-train-accuracy", default=0.8, type=float)
@click.option("--min-test-accuracy", default=0.8, type=float)
@click.option("--fail-on-accuracy-quality-gates", is_flag=True, default=False)
@click.option("--only-inference", is_flag=True, default=False)
def main(
    no_cache: bool,
    no_drop_na: bool,
    no_normalize: bool,
    drop_columns: Optional[str],
    test_size: float,
    min_train_accuracy: float,
    min_test_accuracy: float,
    fail_on_accuracy_quality_gates: bool,
    only_inference: bool,
):
    """Main entry point for pipeline execution."""

    pipeline_args = {}

    if no_cache:
        pipeline_args["enable_cache"] = False

    # -------------------------
    # TRAINING PIPELINE
    # -------------------------
    if not only_inference:
        run_args_train = {
            "drop_na": not no_drop_na,
            "normalize": not no_normalize,
            "test_size": test_size,
            "min_train_accuracy": min_train_accuracy,
            "min_test_accuracy": min_test_accuracy,
            "fail_on_accuracy_quality_gates": fail_on_accuracy_quality_gates,
        }

        if drop_columns:
            run_args_train["drop_columns"] = drop_columns.split(",")

        pipeline_args["config_path"] = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "configs",
            "train_config.yaml",
        )

        pipeline_args["run_name"] = (
            f"e2e_use_case_training_run_{dt.now().strftime('%Y_%m_%d_%H_%M_%S')}"
        )

        e2e_use_case_training.with_options(**pipeline_args)(**run_args_train)
        logger.info("Training pipeline finished successfully!")

    # -------------------------
    # DEPLOYMENT PIPELINE
    # -------------------------
    pipeline_args["config_path"] = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "configs",
        "deployer_config.yaml",
    )

    pipeline_args["run_name"] = (
        f"e2e_use_case_deployment_run_{dt.now().strftime('%Y_%m_%d_%H_%M_%S')}"
    )

    e2e_use_case_deployment.with_options(**pipeline_args)()

    # -------------------------
    # BATCH INFERENCE (CI-SAFE)
    # -------------------------
    if IS_CI:
        logger.warning("CI mode detected → Skipping batch inference pipeline.")
        return

    pipeline_args["config_path"] = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "configs",
        "inference_config.yaml",
    )

    pipeline_args["run_name"] = (
        f"e2e_use_case_batch_inference_run_{dt.now().strftime('%Y_%m_%d_%H_%M_%S')}"
    )

    e2e_use_case_batch_inference.with_options(**pipeline_args)()


if __name__ == "__main__":
    main()
