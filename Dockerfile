FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt setup.py MLproject ./
COPY app ./app
COPY src ./src
COPY dashboard ./dashboard
COPY data ./data
COPY tests ./tests
COPY params.yaml README.md Procfile tox.ini ./

RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && python src/data/load_data.py --config params.yaml \
    && python src/data/split_data.py --config params.yaml \
    && python src/models/train_model.py --config params.yaml --n-trials 10

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
