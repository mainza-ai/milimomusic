#!/usr/bin/env bash
# ==============================================================================
# Milimo Music — 1-Click Docker Startup Script
# Automatically selects GPU (NVIDIA Container Toolkit) or CPU fallback compose.
# ==============================================================================
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

echo "========================================================"
echo "          MILIMO MUSIC — DOCKER LAUNCHER                "
echo "========================================================"

# Check docker is running
if ! command -v docker &> /dev/null; then
    echo "❌ Error: Docker is not installed or not in PATH."
    echo "   Please install Docker Desktop or Docker Engine: https://docs.docker.com/get-docker/"
    exit 1
fi

if ! docker info &> /dev/null; then
    echo "❌ Error: Docker daemon is not running. Please start Docker."
    exit 1
fi

COMPOSE_FILE="docker-compose.yml"
HAS_NVIDIA=false

if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    if docker info 2>/dev/null | grep -iq "nvidia"; then
        HAS_NVIDIA=true
    fi
fi

if [ "$HAS_NVIDIA" = true ]; then
    echo "⚡ Detected NVIDIA GPU with Docker GPU support. Using GPU profile."
    COMPOSE_FILE="docker-compose.yml"
else
    echo "💻 No NVIDIA container runtime detected. Using CPU / Apple Silicon / standard profile."
    COMPOSE_FILE="docker-compose.cpu.yml"
fi

echo "🚀 Starting Milimo Music container via $COMPOSE_FILE..."
docker compose -f "$COMPOSE_FILE" up -d --build

echo "⏳ Waiting for Milimo Music backend to become healthy..."
RETRIES=45
COUNT=0
HEALTHY=false

while [ $COUNT -lt $RETRIES ]; do
    if curl -s -f http://localhost:8000/health > /dev/null 2>&1; then
        HEALTHY=true
        break
    fi
    printf "."
    sleep 2
    COUNT=$((COUNT + 1))
done
echo ""

if [ "$HEALTHY" = true ]; then
    echo "========================================================"
    echo "✅ Milimo Music is up and running!"
    echo "   URL: http://localhost:8000"
    echo "   Logs: docker compose -f $COMPOSE_FILE logs -f"
    echo "   Stop: docker compose -f $COMPOSE_FILE down"
    echo "========================================================"
else
    echo "⚠️ Backend health check timed out. Checking container logs:"
    docker compose -f "$COMPOSE_FILE" logs --tail 30
    echo "========================================================"
    echo "Run 'docker compose -f $COMPOSE_FILE logs -f' to inspect."
fi
