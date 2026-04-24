FROM python:3.12-slim

RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install -r requirements.txt \
    && pip install fastapi uvicorn

ENV PYTHONPATH=/app

RUN mkdir -p /app/mlruns

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
