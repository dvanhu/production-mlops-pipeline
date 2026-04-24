# Production MLOps Pipeline
[![MLOps Pipeline](https://github.com/dvanhu/production-mlops-pipeline/actions/workflows/mlops-pipeline.yml/badge.svg)](https://github.com/dvanhu/production-mlops-pipeline/actions/workflows/mlops-pipeline.yml)
[![Docker Pulls](https://img.shields.io/docker/pulls/dvanhu/mlops-pipeline?label=Docker%20Pulls&logo=docker&logoColor=white&color=2496ED)](https://hub.docker.com/r/dvanhu/mlops-pipeline)
[![Docker Image Size](https://img.shields.io/docker/image-size/dvanhu/mlops-pipeline/latest?label=Image%20Size&logo=docker&logoColor=white&color=2496ED)](https://hub.docker.com/r/dvanhu/mlops-pipeline)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![ZenML](https://img.shields.io/badge/ZenML-Orchestration-9F2BFF?logo=zenml&logoColor=white)](https://zenml.io/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking%20%26%20Registry-0194E2?logo=mlflow&logoColor=white)](https://mlflow.org/)

---

## Overview

This repository implements a full model lifecycle pipeline structured around three operationally independent ZenML pipelines: training, deployment, and batch inference. The system covers data ingestion through model promotion, registry-driven deployment, and monitored batch scoring with Evidently drift detection.

The design premise is that each stage of the ML lifecycle — retraining, deploying, and scoring — should be independently triggerable without side effects on other stages. The training pipeline produces a candidate model and makes a data-driven promotion decision. The deployment pipeline takes whatever model the registry designates as `Production` and serves it. The batch inference pipeline consumes the serving endpoint and runs drift detection against the training reference dataset before generating predictions.

MLflow tracks all experiments using a local file-based backend (`file:./mlruns`), with no dependency on the ZenML experiment tracker integration. The ZenML local stack coordinates artifact lineage, step execution, and the Evidently data validator. Docker provides a reproducible execution environment, and GitHub Actions runs the pipeline on every push to `main`, with batch inference gated behind a CI environment check to keep workflow execution time bounded.

---

## Architecture

The three pipelines share preprocessing logic and interact through the MLflow model registry. The diagram below shows the inter-pipeline relationships and the flow of artifacts between stages.

<p align="center">
  <img src=".assets/00_pipelines_composition.png" alt="Pipeline Composition" width="85%"/>
</p>

The training pipeline writes a registered model entry to MLflow. The deployment pipeline reads from that registry entry by stage (`Production`) and exposes the model for inference. The batch inference pipeline resolves the deployment endpoint, applies drift detection, and writes scored predictions alongside a drift report.

---

## Pipeline Breakdown

### ETL

<p align="center">
  <img src=".assets/01_etl.png" alt="ETL Pipeline" width="80%"/>
</p>

The ETL step ingests raw data, validates schema, applies preprocessing transformations, and produces versioned train/validation/test splits as ZenML artifacts. The preprocessing logic defined here is the single implementation reused by the batch inference pipeline. This is the primary mechanism for avoiding training-serving skew: there is one transformation function, not two.

---

### Hyperparameter Tuning

<p align="center">
  <img src=".assets/02_hp.png" alt="Hyperparameter Tuning" width="80%"/>
</p>

A search is conducted over a parameter space defined in the pipeline YAML config. Each trial is logged as a child MLflow run nested under the parent training run. The best configuration by validation metric is selected and passed to the training step. All trial parameters, metrics, and fitted estimators are stored in MLflow for later comparison.

---

### Training

<p align="center">
  <img src=".assets/03_train.png" alt="Training Pipeline" width="80%"/>
</p>

A model is trained on the full training split using the optimal hyperparameters from the tuning step. The run logs parameters, evaluation metrics, the dataset hash, and the serialized estimator to MLflow. The model is registered in the MLflow model registry under a versioned entry tagged with the run ID, making it traceable back to the exact training execution that produced it.

---

### Promotion

<p align="center">
  <img src=".assets/04_promotion.png" alt="Promotion Pipeline" width="80%"/>
</p>

The promotion step compares the newly trained candidate model against the model currently in the `Production` stage of the MLflow registry, evaluated on the held-out test set. If the candidate meets or exceeds the baseline on the configured primary metric, it is transitioned to `Production` in the registry. If it does not, the existing production model is retained and the run is logged with the comparison result. No manual approval is required in standard retraining cycles.

---

### Deployment

<p align="center">
  <img src=".assets/05_deployment.png" alt="Deployment Pipeline" width="80%"/>
</p>

The deployment pipeline queries the MLflow registry for the current `Production`-stage model and deploys it to a local MLflow model server. The deployment step is decoupled from the training pipeline: a model can be redeployed without triggering a retraining run, and training can proceed without touching the serving layer. The active deployment is registered in the ZenML stack context so that the batch inference pipeline can resolve the endpoint without hardcoded references.

---

### Batch Inference

<p align="center">
  <img src=".assets/06_batch_inference.png" alt="Batch Inference Pipeline" width="80%"/>
</p>

The batch inference pipeline applies the same preprocessing transformations as the training ETL step, then loads the production model to generate predictions. Before scoring, an Evidently drift report is computed by comparing the incoming batch against the training reference dataset stored as a ZenML artifact. The report is saved as an HTML artifact and logged to the associated MLflow run. If overall drift exceeds the configured threshold, the run is tagged `drift_detected=true` in MLflow.

**CI environment handling:** Batch inference is skipped in GitHub Actions using an explicit environment check:

```python
if os.getenv("CI") == "true":
    print("Skipping batch inference in CI environment.")
    return
```

This is intentional. Batch inference depends on a live deployment endpoint that is not available in the ephemeral CI environment. Skipping this step in CI allows the workflow to validate the training and deployment pipelines without requiring a running inference server.

---

## Tech Stack

| Tool | Version | Role |
|---|---|---|
| ZenML | Latest stable | Pipeline orchestration, artifact versioning, local stack management |
| MLflow | Latest stable | Experiment tracking (local file backend), model registry, run comparison |
| Evidently | Latest stable | Data drift detection and quality reporting in batch inference |
| Scikit-learn | Latest stable | Model training, hyperparameter search, preprocessing transformations |
| Docker | — | Containerized execution environment; image published to Docker Hub |
| GitHub Actions | — | CI/CD: pipeline execution and Docker image publication on push to `main` |
| Python | 3.12 | Runtime environment |

---

## Docker Usage

The pipeline image is published to Docker Hub and is the canonical environment for running the pipeline outside of a local Python setup.

**Pull the image:**

```bash
docker pull dvanhu/mlops-pipeline
```

**Run the training pipeline:**

```bash
docker run --rm \
  -v $(pwd)/mlruns:/app/mlruns \
  dvanhu/mlops-pipeline
```

The volume mount (`-v $(pwd)/mlruns:/app/mlruns`) is required to persist MLflow tracking data across container runs. Without it, all experiment logs, registered models, and artifacts are lost when the container exits. Mount this directory to a stable host path and use the same path when launching the MLflow UI.

**With a custom configuration file:**

```bash
docker run --rm \
  -v $(pwd)/mlruns:/app/mlruns \
  -v $(pwd)/configs/custom.yaml:/app/configs/training_config.yaml \
  dvanhu/mlops-pipeline
```

**Key Dockerfile decisions:**

- Dependencies installed from `requirements.txt` only — no `pip install .` or editable installs
- `PYTHONPATH=/app` set as an environment variable so module imports resolve correctly without a package installation step
- `/app/mlruns` created at build time as the default MLflow artifact directory; overridden by the volume mount at runtime

---

## Experiment Tracking

MLflow is configured with a local file-based tracking URI:

```python
mlflow.set_tracking_uri("file:./mlruns")
```

This means all tracking data is written to the `./mlruns` directory relative to the working directory. There is no dependency on a remote tracking server or the ZenML MLflow experiment tracker integration. This design simplifies local setup and CI execution at the cost of not having a shared tracking server for team environments.

Each training pipeline run creates a parent MLflow run. Hyperparameter tuning trials are logged as nested child runs under the parent, which allows all trials within a single training cycle to be compared in the MLflow UI without cross-contamination between runs.

The following are logged per training run:

- Hyperparameter values and the search configuration
- Per-fold validation metrics from the tuning step
- Final test set evaluation metrics (as defined in the pipeline config)
- The fitted model artifact, serialized using the `sklearn` MLflow flavor
- Dataset hash and split sizes for traceability

**To view the tracking UI locally:**

```bash
mlflow ui --backend-store-uri ./mlruns
```

Then navigate to `http://localhost:5000`.

---

## Data Drift Detection

Evidently is integrated into the batch inference pipeline via the ZenML Evidently validator step. The drift report is computed against a reference dataset — the training split saved as a ZenML artifact during the ETL step — each time a batch scoring run executes.

The report covers:

- Per-feature distribution shift, using statistical tests matched to each feature's data type (chi-squared for categorical, KS test for continuous)
- Dataset-level drift summary with an overall verdict
- Missing value rates and out-of-range value counts per feature

The HTML report is logged to the batch inference run in MLflow. If the dataset-level drift score exceeds the configured threshold, the run is tagged `drift_detected=true`. The pipeline does not halt on drift detection by default; it logs and continues. Halting behavior can be configured in the pipeline YAML.

---

## Project Structure

```
production-mlops-pipeline/
├── .assets/                    # Architecture diagrams used in this README
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions workflow: pipeline run + Docker publish
├── .zen/                       # ZenML local stack configuration
├── configs/                    # YAML configuration files for pipeline runs
│                               #   (hyperparameter space, promotion thresholds, drift thresholds)
├── pipelines/                  # ZenML pipeline definitions; each file assembles a DAG from steps
├── steps/                      # Individual ZenML steps
│                               #   (ingest, preprocess, tune, train, evaluate, promote,
│                               #    deploy, drift_check, batch_score)
├── utils/                      # Shared utilities
│                               #   (preprocessing functions, metric helpers, registry clients)
├── .dockerignore
├── .gitignore
├── Dockerfile                  # Container build: python:3.12-slim, requirements.txt, PYTHONPATH=/app
├── Makefile                    # Convenience targets for local development
├── api.py                      # FastAPI inference endpoint (optional serving layer)
├── requirements.txt            # Pinned dependencies: zenml[jupyter,server], mlflow, evidently
└── run.py                      # CLI entry point for local pipeline execution
```

---

## Running Locally

**1. Clone the repository:**

```bash
git clone https://github.com/dvanhu/production-mlops-pipeline.git
cd production-mlops-pipeline
```

**2. Install dependencies:**

```bash
pip install -r requirements.txt
```

**3. Initialize ZenML and configure the local stack:**

```bash
zenml init
zenml integration install mlflow evidently scikit-learn -y

# Register components
zenml experiment-tracker register mlflow_tracker --flavor=mlflow
zenml model-deployer register mlflow_deployer --flavor=mlflow
zenml data-validator register evidently_validator --flavor=evidently

# Assemble and activate the stack
zenml stack register local_stack \
  -a default \
  -o default \
  -e mlflow_tracker \
  -d mlflow_deployer \
  -dv evidently_validator

zenml stack set local_stack
```

**4. Run a pipeline:**

```bash
# Training pipeline (ETL → tuning → training → promotion)
python run.py --pipeline training

# Deployment pipeline
python run.py --pipeline deployment

# Batch inference pipeline
python run.py --pipeline batch_inference
```

Running `python run.py` without flags executes the training pipeline by default.

**5. View experiment results:**

```bash
mlflow ui --backend-store-uri ./mlruns
```

---

## CI/CD Pipeline

The GitHub Actions workflow file (`.github/workflows/ci.yml`) runs on every push to `main`. The workflow performs the following steps in sequence:

1. **Checkout and environment setup** — Python 3.12, pip cache restore, dependency installation from `requirements.txt`
2. **ZenML stack initialization** — Initializes ZenML with the local stack; registers MLflow and Evidently integrations
3. **Training pipeline execution** — Runs the full training pipeline (ETL, hyperparameter tuning, training, promotion); the workflow step fails and blocks subsequent steps if any ZenML step exits with an error
4. **Batch inference skip** — The batch inference pipeline detects `CI=true` in the environment and exits early, since no deployment endpoint is available in the CI runner
5. **Docker image build** — Builds the image from `Dockerfile` at the current commit SHA
6. **Docker Hub publish** — Authenticates using `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` repository secrets and pushes the image tagged with both the commit SHA and `latest`

The Docker image published on each successful CI run is the versioned artifact corresponding to the code state that passed the pipeline validation. This means every `dvanhu/mlops-pipeline` image tag maps to a specific commit that successfully completed training and promotion.

---

## Key Features

- **Single preprocessing implementation.** The transformation logic in the ETL step is the same code path executed in batch inference. There is no separate preprocessing module for serving.
- **Registry-gated promotion.** No model reaches the `Production` stage without passing a quantitative comparison against the incumbent. The promotion decision is logged and traceable in MLflow.
- **Decoupled pipeline stages.** Training, deployment, and batch inference are independent pipelines. Each can be triggered, paused, or rerun without affecting the others.
- **Local file-based MLflow tracking.** No tracking server infrastructure is required for local development or CI. The `./mlruns` directory is the complete experiment store.
- **Inline drift monitoring.** Evidently runs within the batch inference pipeline, not as a separate monitoring service. Drift results are co-located with inference run metadata in MLflow.
- **Versioned Docker image per CI run.** Every successful build produces a tagged image on Docker Hub, providing a reproducible execution environment for each code version.
- **Externalized configuration.** Pipeline behavior (hyperparameter space, promotion threshold, drift threshold) is defined in YAML config files. Retraining with different parameters requires no code changes.

---

## Author

**dvanhu**

- GitHub: [github.com/dvanhu](https://github.com/dvanhu)
- Docker Hub: [hub.docker.com/r/dvanhu/mlops-pipeline](https://hub.docker.com/r/dvanhu/mlops-pipeline)

---

## License

See [LICENSE](LICENSE) for details.
