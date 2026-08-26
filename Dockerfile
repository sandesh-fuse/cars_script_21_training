FROM python:3.11-slim

# System deps. libgomp is required by XGBoost runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (cache layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source, then artifacts. Artifacts are baked into the image (not
# mounted as a volume) so it's self-contained for deployment — the
# docker-compose.yml volume-mount block is commented out to match.
COPY preprocessor.py shap_dollar_helper.py ./
COPY app/ ./app/
COPY ./artifacts/ /app/artifacts/

# Default artifacts dir; can be overridden by env
ENV ARTIFACTS_DIR=/app/artifacts
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Healthcheck hits /healthz
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request, sys; \
        r = urllib.request.urlopen('http://localhost:8000/healthz', timeout=3); \
        sys.exit(0 if r.status == 200 else 1)" || exit 1

# Single worker is fine; XGBoost predict and SHAP are CPU-bound already.
# For higher throughput, increase --workers (each worker loads its own copy
# of the artifacts, so RAM scales linearly).
# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--log-config", "app/log_config.json"]