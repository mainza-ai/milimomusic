"""Sidecar Virtual Environment & Engine Isolation Manager.

Isolates conflicting heavy neural stacks (e.g., MuLaCover, Neural SVC, Wan 2.1)
into dedicated Python virtual environments under backend/engines/<id>/.venv
managed via uv, preventing dependency hell and version collisions.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import logging
import os
import subprocess
import sys

logger = logging.getLogger("milimo.sidecar.engine_manager")


class SidecarEngineManager:
    """Manages uv-isolated sidecar runtime environments."""

    def __init__(self, engines_base_dir: str = "backend/engines"):
        self.base_dir = Path(engines_base_dir).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def get_engine_dir(self, engine_id: str) -> Path:
        return self.base_dir / engine_id

    def get_venv_python(self, engine_id: str) -> Path:
        engine_dir = self.get_engine_dir(engine_id)
        if sys.platform == "win32":
            return engine_dir / ".venv" / "Scripts" / "python.exe"
        return engine_dir / ".venv" / "bin" / "python"

    def is_engine_ready(self, engine_id: str) -> bool:
        return self.get_venv_python(engine_id).is_file()

    def ensure_sidecar(
        self,
        engine_id: str,
        requirements: Optional[List[str]] = None,
        python_version: Optional[str] = None,
    ) -> Path:
        """Create an isolated uv virtual environment and install dependencies.

        Args:
            engine_id: Unique identifier for the sidecar engine (e.g. 'mulacover', 'svc').
            requirements: List of pip packages required by the engine.
            python_version: Optional Python version to target (e.g. '3.11', '3.12').

        Returns:
            Path to the sidecar Python executable.
        """
        engine_dir = self.get_engine_dir(engine_id)
        engine_dir.mkdir(parents=True, exist_ok=True)
        venv_path = engine_dir / ".venv"
        py_exec = self.get_venv_python(engine_id)

        if not py_exec.is_file():
            logger.info(f"Initializing uv virtualenv for sidecar '{engine_id}' at {venv_path}...")
            cmd = ["uv", "venv", str(venv_path)]
            if python_version:
                cmd.extend(["--python", python_version])
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                logger.error(f"Failed to create uv venv: {res.stderr}")
                raise RuntimeError(f"uv venv creation failed: {res.stderr}")

        if requirements and len(requirements) > 0:
            logger.info(f"Installing {len(requirements)} packages for sidecar '{engine_id}'...")
            pip_cmd = ["uv", "pip", "install", "--python", str(py_exec)] + requirements
            res = subprocess.run(pip_cmd, capture_output=True, text=True)
            if res.returncode != 0:
                logger.error(f"Failed to install sidecar packages: {res.stderr}")
                raise RuntimeError(f"uv pip install failed for {engine_id}: {res.stderr}")

        logger.info(f"Sidecar '{engine_id}' ready at {py_exec}")
        return py_exec

    def run_command(
        self,
        engine_id: str,
        script_or_module: str,
        args: Optional[List[str]] = None,
        cwd: Optional[Path] = None,
        timeout: Optional[float] = 300.0,
    ) -> subprocess.CompletedProcess:
        """Execute a Python command inside the isolated sidecar environment."""
        py_exec = self.get_venv_python(engine_id)
        if not py_exec.is_file():
            raise FileNotFoundError(f"Sidecar environment not initialized for '{engine_id}'.")

        cmd = [str(py_exec), script_or_module] + (args or [])
        return subprocess.run(
            cmd,
            cwd=str(cwd or self.get_engine_dir(engine_id)),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=True,
        )


sidecar_manager = SidecarEngineManager()
