FROM python:3.11-slim

WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY migrations ./migrations
COPY alembic.ini .
COPY config ./config
COPY models ./models

# Pre-download the sentence-transformers embedding model at build time instead of
# lazily on first request - the lazy-load path hit HuggingFace's unauthenticated rate
# limit on Cloud Run (every scale-to-zero cold start is a fresh container re-downloading
# the same model) and crashed the first deploy. Baking it into the image removes both
# the network dependency and the rate-limit exposure from the request path entirely.
RUN python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

EXPOSE 8000

# Cloud Run injects the actual listen port via $PORT (defaults to 8080, not 8000) and
# healthchecks the container on it - a hardcoded --port 8000 here caused the first
# deploy to fail its startup healthcheck even though the app itself was fine.
ENV PORT=8000
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT}
