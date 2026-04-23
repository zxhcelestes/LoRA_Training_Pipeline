FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl build-essential libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p outputs logs

RUN useradd -m -u 1000 pipeline && chown -R pipeline:pipeline /app
USER pipeline

EXPOSE 8000

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LOG_LEVEL=INFO

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--log-level", "info"]
