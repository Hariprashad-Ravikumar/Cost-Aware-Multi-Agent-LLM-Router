FROM python:3.11-slim

WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY migrations ./migrations
COPY alembic.ini .
COPY config ./config
COPY models ./models

EXPOSE 8000

# Cloud Run injects the actual listen port via $PORT (defaults to 8080, not 8000) and
# healthchecks the container on it - a hardcoded --port 8000 here caused the first
# deploy to fail its startup healthcheck even though the app itself was fine.
ENV PORT=8000
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT}
