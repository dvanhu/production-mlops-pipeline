FROM python:3.12-slim

# System deps (optional but safer for sklearn/pandas)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working dir
WORKDIR /app

# Copy files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install -r requirements.txt

# Install project as module (VERY IMPORTANT)
RUN pip install .

# Set python path
ENV PYTHONPATH=/app

# Create MLflow storage
RUN mkdir -p /app/mlruns

# Default command
CMD ["python", "run.py"]
