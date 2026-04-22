# Apache Software License 2.0
# Copyright (c) ZenML GmbH 2026

from typing import Optional
from typing_extensions import Annotated

from zenml import ArtifactConfig, get_step_context, step
from zenml.client import Client
from zenml.integrations.mlflow.services.mlflow_deployment import MLFlowDeploymentService
from zenml.integrations.mlflow.steps.mlflow_deployer import (
    mlflow_model_registry_deployer_step,
)
from zenml.logger import get_logger

logger = get_logger(__name__)


@step
def deployment_deploy() -> (
    Annotated[
        Optional[MLFlowDeploymentService],
        ArtifactConfig(name="mlflow_deployment", is_deployment_artifact=True),
    ]
):
    """
    Deployment step using MLflow Model Registry.

    This step deploys a model if a valid MLflow registry version exists.
    Otherwise, it safely skips deployment.
    """

    # Only deploy if using local orchestrator
    if Client().active_stack.orchestrator.flavor == "local":

        model = get_step_context().model

        # ✅ SAFE ACCESS (no crash)
        registry_model_version = model.run_metadata.get("model_registry_version")

        # ✅ Handle missing MLflow registry
        if registry_model_version is None:
            logger.warning("Skipping deployment — no MLflow registry model found")
            return None

        logger.info(
            f"Deploying model '{model.name}' with registry version '{registry_model_version}'"
        )

        # ✅ Deploy model
        deployment_service = mlflow_model_registry_deployer_step.entrypoint(
            registry_model_name=model.name,
            registry_model_version=registry_model_version,
            replace_existing=True,
        )

    else:
        logger.warning("Skipping deployment — non-local orchestrator")
        deployment_service = None

    return deployment_service
