# Multi-stage Dockerfile for Milimo Music (Single Unified Container)
# Stage 1: Build Frontend SPA
FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# Stage 2: Production Python Backend & Web DAW Runtime
FROM python:3.11-slim
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    MILIMO_IN_DOCKER=1 \
    PYTHONPATH=/app/backend:/app/muscriptor

WORKDIR /app

# System audio, video, DSP & build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    fluidsynth \
    libgl1 \
    libglib2.0-0 \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir --upgrade pip uv && \
    uv pip install --system --no-cache -r /app/backend/requirements.txt

# Copy backend, neural transcription, legacy heartlib
COPY backend /app/backend
COPY muscriptor /app/muscriptor
COPY heartlib /app/heartlib

# Copy compiled frontend from Stage 1 into /app/frontend/dist
COPY --from=frontend-builder /app/frontend/dist /app/frontend/dist

# Persistent data directories
RUN mkdir -p /app/data /app/data/covers /app/generated_audio /app/generated_midi /app/models

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
