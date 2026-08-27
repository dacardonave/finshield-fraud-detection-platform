# FinShield Fraud Detection Platform - container image
#
# One image, two possible entrypoints. The API (api/main.py) and the
# Streamlit demo (streamlit_app.py) both just need the trained model and
# the same Python dependencies from requirements.txt - there's no reason
# to build and maintain two separate images. Which one actually runs is
# decided by the `command:` each service sets in docker-compose.yml; the
# CMD below is only the default if the image is run standalone.
#
# Only what's needed to *serve* the already-trained model is copied in -
# not notebooks, tests, or the raw dataset. Training happens outside the
# container (`python -m src.train`) and its output (models/model.joblib,
# models/model_metadata.json) is what gets shipped here.

FROM python:3.13-slim

WORKDIR /app

# Dependencies first, so this (slow) layer is cached across code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY api/ api/
COPY streamlit_app.py .
COPY models/model.joblib models/model.joblib
COPY models/model_metadata.json models/model_metadata.json

# Run as a non-root user rather than the container default (root).
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000 8501

# Cloud platforms like Render assign a port at runtime via the $PORT
# env var and expect the app to listen there - they don't let you pick
# a fixed one. Locally (docker run / docker compose), $PORT is unset,
# so this falls back to 8000, same as before. Explicit "sh -c" (JSON
# form) rather than bare shell-form CMD, so Docker forwards stop
# signals correctly while still letting ${PORT:-8000} expand.
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
