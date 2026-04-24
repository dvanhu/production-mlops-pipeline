import os

# Detect CI environment (GitHub Actions automatically sets CI=true)
IS_CI = os.getenv("CI", "false").lower() == "true"


def main(...):

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
