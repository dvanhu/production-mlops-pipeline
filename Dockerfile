FROM python:3.12-slim

# -----------------------------
# System dependencies
# -----------------------------
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------
# Set working directory
# -----------------------------
WORKDIR /app

# -----------------------------
# Copy only requirements first (cache optimization)
# -----------------------------
COPY requirements.txt .

# -----------------------------
# Install dependencies
# -----------------------------
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# -----------------------------
# Copy project files
# -----------------------------
COPY . .

# -----------------------------
# Fix Python imports
# -----------------------------
ENV PYTHONPATH=/app

# -----------------------------
# MLflow storage (persistent mount compatible)
# -----------------------------
RUN mkdir -p /app/mlruns

# -----------------------------
# ZenML + Run Pipeline
# -----------------------------
CMD ["bash", "-c", "zenml init || true && python run.py"]
