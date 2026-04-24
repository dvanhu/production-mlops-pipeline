FROM python:3.12-slim

# System deps
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy everything
COPY . .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install -r requirements.txt

ENV PYTHONPATH=/app

# Create MLflow dir
RUN mkdir -p /app/mlruns

CMD ["python", "run.py"]
