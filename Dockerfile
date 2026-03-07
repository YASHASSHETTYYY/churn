FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt setup.py ./
COPY src ./src
COPY dashboard ./dashboard
COPY data ./data
COPY tests ./tests
COPY params.yaml README.md app.py Procfile tox.ini ./

RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && python src/data/load_data.py --config params.yaml \
    && python src/data/split_data.py --config params.yaml \
    && python src/models/train_model.py --config params.yaml --n-trials 10

EXPOSE 8000

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
