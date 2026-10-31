# Serves the FastAPI backend and the static Leaflet frontend.
# Build from the repo root: docker build -f app/Dockerfile -t dengue-map .
FROM python:3.11-slim
WORKDIR /srv
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY app/requirements.txt app/requirements.txt
RUN pip install --no-cache-dir -r app/requirements.txt
COPY app/__init__.py app/__init__.py
COPY app/api app/api
COPY app/web app/web
COPY app/data app/data
EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
