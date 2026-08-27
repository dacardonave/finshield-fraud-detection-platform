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

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
