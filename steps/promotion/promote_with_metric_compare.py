from zenml import step, get_step_context, Model
from zenml.logger import get_logger
from utils import promote_in_model_registry

logger = get_logger(__name__)

@step
def promote_with_metric_compare(
    latest_metric: float,
    current_metric: float,
    mlflow_model_name: str,
    target_env: str,
) -> None:

    should_promote = True

    # Get model version numbers
    latest_version = get_step_context().model
    current_version = Model(name=latest_version.name, version=target_env)

    try:
        current_version_number = current_version.number
    except KeyError:
        current_version_number = None

    if current_version_number is None:
        logger.info("No current model version found - promoting latest")
    else:
        logger.info(
            f"Latest model metric={latest_metric:.6f}\n"
            f"Current model metric={current_metric:.6f}"
        )
        if latest_metric >= current_metric:
            logger.info("Latest model is better → promoting")
        else:
            logger.info("Current model is better → keeping current")
            should_promote = False

    # ✅ FIXED: INSIDE FUNCTION
    if should_promote:
        model = get_step_context().model
        model.set_stage(stage=target_env, force=True)
        logger.info(f"Promoted to '{target_env}'")

        latest_version_model_registry_number = latest_version.run_metadata.get(
            "model_registry_version"
        )

        if latest_version_model_registry_number is None:
            logger.warning("No MLflow registry version found → skipping")
            promoted_version = "N/A"
        else:
            if current_version_number is None:
                current_version_model_registry_number = (
                    latest_version_model_registry_number
                )
            else:
                current_version_model_registry_number = current_version.run_metadata.get(
                    "model_registry_version"
                )

            promote_in_model_registry(
                latest_version=latest_version_model_registry_number,
                current_version=current_version_model_registry_number,
                model_name=mlflow_model_name,
                target_env=target_env.capitalize(),
            )

            promoted_version = latest_version_model_registry_number

    else:
        promoted_version = current_version.run_metadata.get(
            "model_registry_version", "N/A"
        )

    logger.info(
        f"Current model version in `{target_env}` is `{promoted_version}`"
    )
