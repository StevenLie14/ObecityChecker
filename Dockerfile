FROM python:3.12-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir gunicorn==23.00


FROM python:3.12-slim

WORKDIR /app

COPY --from=builder /usr/local /usr/local

COPY . .

CMD ["gunicorn", "-c", "gunicorn.conf.py", "backend:app"]
