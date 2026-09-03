#!/usr/bin/env bash
set -e

# Detect dedicated conda environment
if [ -d "/opt/miniconda3/envs/milimomusic" ]; then
    PYTHON_EXEC="/opt/miniconda3/envs/milimomusic/bin/python"
elif command -v conda &>/dev/null && conda info --envs | grep -q "milimomusic"; then
    PYTHON_EXEC="conda run -n milimomusic python"
else
    PYTHON_EXEC="python3"
fi

echo "==> Starting Milimo Music Backend with: $PYTHON_EXEC"
export PYTHONPATH="backend:muscriptor:${PYTHONPATH:-}"

exec $PYTHON_EXEC -m uvicorn app.main:app --host 0.0.0.0 --port 8000 "$@"
