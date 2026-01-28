"""
FineTuningService - Training job management for HeartMuLa style expansion.

Handles:
- Dataset creation and preprocessing (audio → tokens)
- Training job queue with status tracking
- Checkpoint management
"""

import os
import json
import uuid
import asyncio
import subprocess
import logging
from pathlib import Path
from typing import List, Optional, Literal
from dataclasses import dataclass, asdict, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    DRAFT = "draft"
    PREPROCESSING = "preprocessing"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class AudioFile:
    """Represents an audio file in a dataset."""
    filename: str
    caption: str
    duration_seconds: Optional[float] = None
    preprocessed: bool = False


@dataclass
class Dataset:
    """Training dataset with audio files and captions."""
    id: str
    name: str
    styles: List[str]
    audio_files: List[AudioFile] = field(default_factory=list)
    status: str = "draft"
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> dict:
        d = asdict(self)
        d["audio_files"] = [asdict(f) for f in self.audio_files]
        return d
    
    @classmethod
    def from_dict(cls, data: dict) -> "Dataset":
        audio_files = [AudioFile(**f) for f in data.pop("audio_files", [])]
        return cls(audio_files=audio_files, **data)


@dataclass
class TrainingConfig:
    """Configuration for a training job."""
    dataset_id: str
    method: Literal["lora", "full"] = "lora"
    epochs: int = 3
    learning_rate: float = 1e-4
    batch_size: int = 2
    # LoRA-specific
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    # Full fine-tune specific
    gradient_checkpointing: bool = True


@dataclass
class TrainingJob:
    """Represents a training job."""
    id: str
    dataset_id: str
    config: TrainingConfig
    status: str = "queued"
    progress: int = 0
    current_epoch: int = 0
    current_loss: Optional[float] = None
    initial_loss: Optional[float] = None  # First loss value at start of training
    final_loss: Optional[float] = None  # Final loss when training completes
    total_epochs: int = 3
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    checkpoint_id: Optional[str] = None
    error: Optional[str] = None
    message: Optional[str] = None
    dataset_name: Optional[str] = None  # Persists even if dataset is deleted
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> dict:
        d = asdict(self)
        d["config"] = asdict(self.config)
        return d
    
    @classmethod
    def from_dict(cls, data: dict) -> "TrainingJob":
        config_data = data.pop("config", {})
        config = TrainingConfig(**config_data)
        return cls(config=config, **data)


@dataclass
class Checkpoint:
    """Represents a trained model checkpoint."""
    id: str
    name: str
    styles: List[str]
    method: str  # "lora" or "full"
    dataset_id: str
    job_id: str
    created_at: str
    size_bytes: int = 0
    is_active: bool = False
    
    def to_dict(self) -> dict:
        return asdict(self)


class FineTuningService:
    """
    Service for managing fine-tuning workflows.
    
    Storage layout:
    ~/.milimo/
    ├── datasets/{id}/
    │   ├── manifest.json
    │   ├── audio/*.mp3
    │   └── processed/*.pt
    ├── jobs/{id}/
    │   ├── config.yaml
    │   └── logs.txt
    └── checkpoints/{id}/
        ├── meta.json
        └── adapter_model.safetensors
    """
    
    _instance = None
    MINIMUM_AUDIO_FILES = 5
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FineTuningService, cls).__new__(cls)
            cls._instance._init_storage()
        return cls._instance
    
    def _init_storage(self):
        """Initialize storage directories."""
        from .config_manager import ConfigManager
        config = ConfigManager().get_config()
        paths = config.get("paths", {})
        
        self.base_dir = Path(os.path.expanduser(paths.get("datasets_directory", "~/.milimo/datasets"))).parent
        self.datasets_dir = Path(os.path.expanduser(paths.get("datasets_directory", "~/.milimo/datasets")))
        self.checkpoints_dir = Path(os.path.expanduser(paths.get("checkpoints_directory", "~/.milimo/checkpoints")))
        self.jobs_dir = self.base_dir / "jobs"
        
        # Create directories
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory job tracking
        self._active_process: Optional[subprocess.Popen] = None
        self._active_job_id: Optional[str] = None
    
    # --- Dataset Management ---
    
    def create_dataset(self, name: str, styles: List[str]) -> Dataset:
        """Create a new empty dataset."""
        dataset = Dataset(
            id=str(uuid.uuid4()),
            name=name,
            styles=styles
        )
        
        # Create directory structure
        dataset_dir = self.datasets_dir / dataset.id
        (dataset_dir / "audio").mkdir(parents=True, exist_ok=True)
        (dataset_dir / "processed").mkdir(parents=True, exist_ok=True)
        
        # Save manifest
        self._save_dataset(dataset)
        
        logger.info(f"Created dataset '{name}' with id {dataset.id}")
        return dataset
    
    def get_dataset(self, dataset_id: str) -> Optional[Dataset]:
        """Get a dataset by ID."""
        manifest_path = self.datasets_dir / dataset_id / "manifest.json"
        if not manifest_path.exists():
            return None
        
        with open(manifest_path, 'r') as f:
            return Dataset.from_dict(json.load(f))
    
    def list_datasets(self) -> List[Dataset]:
        """List all datasets."""
        datasets = []
        for dataset_dir in self.datasets_dir.iterdir():
            if dataset_dir.is_dir():
                dataset = self.get_dataset(dataset_dir.name)
                if dataset:
                    datasets.append(dataset)
        return datasets
    
    def _save_dataset(self, dataset: Dataset):
        """Save dataset manifest to disk."""
        manifest_path = self.datasets_dir / dataset.id / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(dataset.to_dict(), f, indent=2)
    
    def add_audio_file(self, dataset_id: str, filename: str, caption: str, file_content: bytes) -> AudioFile:
        """Add an audio file to a dataset."""
        dataset = self.get_dataset(dataset_id)
        if not dataset:
            raise ValueError(f"Dataset {dataset_id} not found")
        
        # Save audio file
        audio_path = self.datasets_dir / dataset_id / "audio" / filename
        with open(audio_path, 'wb') as f:
            f.write(file_content)
        
        # Add to manifest
        audio_file = AudioFile(filename=filename, caption=caption)
        dataset.audio_files.append(audio_file)
        
        # Update status to 'ready' if it was 'draft'
        if dataset.status == "draft":
            dataset.status = "ready"
        
        self._save_dataset(dataset)
        
        logger.info(f"Added audio file '{filename}' to dataset {dataset_id}")
        return audio_file
    
    def remove_audio_file(self, dataset_id: str, filename: str) -> bool:
        """Remove an audio file from a dataset."""
        dataset = self.get_dataset(dataset_id)
        if not dataset:
            return False
        
        # Find and remove from manifest
        original_count = len(dataset.audio_files)
        dataset.audio_files = [af for af in dataset.audio_files if af.filename != filename]
        
        if len(dataset.audio_files) == original_count:
            return False  # File not found in manifest
        
        # Delete physical file
        audio_path = self.datasets_dir / dataset_id / "audio" / filename
        if audio_path.exists():
            audio_path.unlink()
        
        self._save_dataset(dataset)
        logger.info(f"Removed audio file '{filename}' from dataset {dataset_id}")
        return True
    
    def update_audio_caption(self, dataset_id: str, filename: str, caption: str) -> bool:
        """Update the caption/lyrics for an audio file."""
        dataset = self.get_dataset(dataset_id)
        if not dataset:
            return False
        
        for audio_file in dataset.audio_files:
            if audio_file.filename == filename:
                audio_file.caption = caption
                self._save_dataset(dataset)
                logger.info(f"Updated caption for '{filename}' in dataset {dataset_id}")
                return True
        
        return False
    
    def validate_dataset(self, dataset_id: str) -> dict:
        """Check if dataset meets minimum requirements."""
        dataset = self.get_dataset(dataset_id)
        if not dataset:
            return {"valid": False, "error": "Dataset not found"}
        
        file_count = len(dataset.audio_files)
        return {
            "valid": file_count >= self.MINIMUM_AUDIO_FILES,
            "file_count": file_count,
            "minimum_required": self.MINIMUM_AUDIO_FILES,
            "message": f"{file_count}/{self.MINIMUM_AUDIO_FILES} audio files"
        }
    
    def update_dataset(self, dataset_id: str, name: Optional[str] = None, styles: Optional[List[str]] = None) -> Optional[Dataset]:
        """Update a dataset's name or styles."""
        dataset = self.get_dataset(dataset_id)
        if not dataset:
            return None
        
        if name is not None:
            dataset.name = name
        if styles is not None:
            dataset.styles = styles
        
        self._save_dataset(dataset)
        logger.info(f"Updated dataset {dataset_id}")
        return dataset
    
    def delete_dataset(self, dataset_id: str) -> bool:
        """Delete a dataset and all its files."""
        import shutil
        dataset_dir = self.datasets_dir / dataset_id
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)
            logger.info(f"Deleted dataset {dataset_id}")
            return True
        return False
    
    def preprocess_dataset(self, dataset_id: str, force: bool = False) -> dict:
        """
        Preprocess a dataset (tokenize audio files).
        
        Args:
            dataset_id: ID of the dataset to preprocess
            force: If True, clear existing processed files first
            
        Returns:
            dict with status and message
        """
        dataset = self.get_dataset(dataset_id)
        if not dataset:
            return {"success": False, "error": "Dataset not found"}
        
        if len(dataset.audio_files) == 0:
            return {"success": False, "error": "Dataset has no audio files"}
        
        dataset_dir = self.datasets_dir / dataset_id
        processed_dir = dataset_dir / "prepared"
        
        # Clear existing processed files if force=True
        if force and processed_dir.exists():
            import shutil
            shutil.rmtree(processed_dir)
            logger.info(f"Cleared existing processed files for dataset {dataset_id}")
        
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Update dataset status
        dataset.status = "preprocessing"
        self._save_dataset(dataset)
        
        try:
            from .config_manager import ConfigManager
            from .training.data_prep import DataPreparator
            import torch
            
            config = ConfigManager().get_config()
            model_path = config.get("paths", {}).get("model_directory", "../heartlib/ckpt")
            
            # Determine device
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
            
            preparator = DataPreparator(
                heartcodec_path=model_path,
                device=device
            )
            
            sample_paths = preparator.prepare_dataset(
                audio_dir=dataset_dir / "audio",
                manifest_path=dataset_dir / "manifest.json",
                output_dir=processed_dir
            )
            
            # Update dataset status
            dataset.status = "processed"
            self._save_dataset(dataset)
            
            logger.info(f"Preprocessed {len(sample_paths)} files for dataset {dataset_id}")
            return {
                "success": True, 
                "processed_count": len(sample_paths),
                "message": f"Successfully processed {len(sample_paths)} audio files"
            }
            
        except Exception as e:
            logger.error(f"Failed to preprocess dataset {dataset_id}: {e}")
            dataset.status = "error"
            self._save_dataset(dataset)
            return {"success": False, "error": str(e)}
    
    # --- Training Jobs ---
    
    def create_training_job(self, config: TrainingConfig) -> TrainingJob:
        """Create a new training job and start training."""
        # Validate dataset
        validation = self.validate_dataset(config.dataset_id)
        if not validation["valid"]:
            raise ValueError(f"Dataset invalid: {validation.get('message', 'Unknown error')}")
        
        # Get dataset name for history persistence
        dataset = self.get_dataset(config.dataset_id)
        dataset_name = dataset.name if dataset else "Unknown Dataset"
        
        job = TrainingJob(
            id=str(uuid.uuid4()),
            dataset_id=config.dataset_id,
            dataset_name=dataset_name,
            config=config,
            total_epochs=config.epochs
        )
        
        # Create job directory
        job_dir = self.jobs_dir / job.id
        job_dir.mkdir(parents=True, exist_ok=True)
        
        # Save job manifest
        self._save_job(job)
        
        logger.info(f"Created training job {job.id}")
        
        # Start training in background
        self._start_training(job)
        
        return job
    
    def _start_training(self, job: TrainingJob):
        """Launch training subprocess."""
        from .config_manager import ConfigManager
        config = ConfigManager().get_config()
        model_path = config.get("paths", {}).get("model_directory", "../heartlib/ckpt")
        
        # Get device
        import torch
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        
        dataset_dir = self.datasets_dir / job.dataset_id
        job_dir = self.jobs_dir / job.id
        
        # Build command
        cmd = [
            "python", "-m", "app.services.training.run_training",
            "--job-id", job.id,
            "--dataset-dir", str(dataset_dir),
            "--output-dir", str(job_dir),
            "--model-path", str(model_path),
            "--epochs", str(job.config.epochs),
            "--learning-rate", str(job.config.learning_rate),
            "--lora-rank", str(job.config.lora_rank),
            "--lora-alpha", str(job.config.lora_alpha),
            "--batch-size", str(job.config.batch_size),
            "--method", job.config.method,
            "--device", device
        ]
        
        logger.info(f"Starting training subprocess: {' '.join(cmd)}")
        
        # Launch subprocess
        log_file = job_dir / "training.log"
        with open(log_file, 'w') as log_f:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=log_f,
                cwd=str(Path(__file__).parent.parent.parent),
                text=True
            )
        
        self._active_process = process
        self._active_job_id = job.id
        
        # Update job status
        job.status = JobStatus.RUNNING.value
        self._save_job(job)
        
        # Start background monitoring
        import threading
        monitor_thread = threading.Thread(
            target=self._monitor_training,
            args=(job.id, process),
            daemon=True
        )
        monitor_thread.start()
    
    def _monitor_training(self, job_id: str, process: subprocess.Popen):
        """Monitor training subprocess and update job progress."""
        import json as json_module
        
        try:
            for line in process.stdout:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    progress = json_module.loads(line)
                    job = self.get_job(job_id)
                    if job:
                        job.current_epoch = progress.get("epoch", 0)
                        loss = progress.get("loss")
                        if loss is not None:
                            job.current_loss = loss
                            # Capture initial loss on first update
                            if job.initial_loss is None:
                                job.initial_loss = loss
                        job.message = progress.get("message")
                        
                        # Use precise percentage if available, else estimate from epoch
                        if "percent" in progress:
                            job.progress = progress["percent"]
                        else:
                            job.progress = int((progress.get("epoch", 0) / job.total_epochs) * 100)
                        
                        status = progress.get("status", "running")
                        if status == "completed":
                            job.status = JobStatus.COMPLETED.value
                            job.progress = 100
                            job.completed_at = datetime.utcnow().isoformat()
                            job.final_loss = progress.get("loss") or job.current_loss  # Store final loss
                            # Create checkpoint
                            self._create_checkpoint_from_job(job)
                            
                            # Refresh style registry to show new trained style
                            from .style_registry import StyleRegistry
                            StyleRegistry().refresh()
                        elif status == "failed":
                            job.status = JobStatus.FAILED.value
                            job.error = progress.get("error", "Training failed")
                            job.completed_at = datetime.utcnow().isoformat()
                        elif status == "preprocessing":
                            job.status = JobStatus.PREPROCESSING.value
                        else:
                            # Set started_at when first transitioning to running
                            if job.status != JobStatus.RUNNING.value and not job.started_at:
                                job.started_at = datetime.utcnow().isoformat()
                            job.status = JobStatus.RUNNING.value
                        
                        self._save_job(job)
                except json_module.JSONDecodeError:
                    continue
            
            # Wait for process to finish
            process.wait()
            
            # Final status update
            job = self.get_job(job_id)
            if job and job.status == JobStatus.RUNNING.value:
                if process.returncode == 0:
                    job.status = JobStatus.COMPLETED.value
                    job.progress = 100
                    self._create_checkpoint_from_job(job)
                else:
                    job.status = JobStatus.FAILED.value
                    # Try to read error from log file
                    log_file = self.jobs_dir / job_id / "training.log"
                    if log_file.exists():
                        try:
                            with open(log_file, 'r') as f:
                                log_content = f.read()
                                # Get last 500 chars of log for error context
                                job.error = log_content[-500:] if len(log_content) > 500 else log_content
                        except:
                            job.error = f"Process exited with code {process.returncode}"
                    else:
                        job.error = f"Process exited with code {process.returncode}"
                self._save_job(job)
                
        except Exception as e:
            logger.error(f"Error monitoring training: {e}")
            job = self.get_job(job_id)
            if job:
                job.status = JobStatus.FAILED.value
                job.error = str(e)
                self._save_job(job)
        finally:
            if self._active_job_id == job_id:
                self._active_process = None
                self._active_job_id = None
    
    def _create_checkpoint_from_job(self, job: TrainingJob):
        """Create a checkpoint from a completed training job."""
        job_dir = self.jobs_dir / job.id
        checkpoint_source = job_dir / "checkpoints" / "checkpoint-final"
        
        if not checkpoint_source.exists():
            logger.warning(f"No checkpoint found for job {job.id}")
            return
        
        # Create checkpoint in checkpoints directory
        dataset = self.get_dataset(job.dataset_id)
        checkpoint_name = f"{dataset.name if dataset else 'custom'}-{job.config.method}"
        checkpoint_id = str(uuid.uuid4())[:8]
        checkpoint_dir = self.checkpoints_dir / f"{checkpoint_name}-{checkpoint_id}"
        
        import shutil
        shutil.copytree(checkpoint_source, checkpoint_dir)
        
        # Add metadata
        meta = {
            "id": f"{checkpoint_name}-{checkpoint_id}",
            "name": checkpoint_name,
            "styles": dataset.styles if dataset else [],
            "method": job.config.method,
            "dataset_id": job.dataset_id,
            "job_id": job.id,
            "created_at": datetime.utcnow().isoformat(),
            "is_active": False
        }
        
        with open(checkpoint_dir / "meta.json", 'w') as f:
            json.dump(meta, f, indent=2)
        
        logger.info(f"Created checkpoint: {checkpoint_name}-{checkpoint_id}")
    
    def get_job(self, job_id: str) -> Optional[TrainingJob]:
        """Get a training job by ID."""
        manifest_path = self.jobs_dir / job_id / "manifest.json"
        if not manifest_path.exists():
            return None
        
        with open(manifest_path, 'r') as f:
            return TrainingJob.from_dict(json.load(f))
    
    def list_jobs(self) -> List[TrainingJob]:
        """List all training jobs."""
        jobs = []
        for job_dir in self.jobs_dir.iterdir():
            if job_dir.is_dir():
                job = self.get_job(job_dir.name)
                if job:
                    jobs.append(job)
        return sorted(jobs, key=lambda j: j.created_at, reverse=True)
    
    def _save_job(self, job: TrainingJob):
        """Save job manifest to disk."""
        manifest_path = self.jobs_dir / job.id / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(job.to_dict(), f, indent=2)
    
    def get_job_logs(self, job_id: str, offset: int = 0) -> List[str]:
        """Get training logs for a job."""
        log_path = self.jobs_dir / job_id / "logs.txt"
        if not log_path.exists():
            return []
        
        with open(log_path, 'r') as f:
            lines = f.readlines()
            return lines[offset:]
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running training job."""
        if self._active_job_id == job_id and self._active_process:
            self._active_process.terminate()
            job = self.get_job(job_id)
            if job:
                job.status = JobStatus.CANCELLED.value
                self._save_job(job)
            return True
        return False
    
    def delete_job(self, job_id: str) -> bool:
        """Delete a training job and its files."""
        import shutil
        job_dir = self.jobs_dir / job_id
        if job_dir.exists():
            # Cancel if running
            if self._active_job_id == job_id:
                self.cancel_job(job_id)
            shutil.rmtree(job_dir)
            logger.info(f"Deleted job {job_id}")
            return True
        return False
    
    # --- Checkpoints ---
    
    def list_checkpoints(self) -> List[Checkpoint]:
        """List all available checkpoints."""
        checkpoints = []
        for ckpt_dir in self.checkpoints_dir.iterdir():
            if ckpt_dir.is_dir():
                meta_path = ckpt_dir / "meta.json"
                if meta_path.exists():
                    try:
                        with open(meta_path, 'r') as f:
                            data = json.load(f)
                            # Backwards compatibility and schema fix
                            if "active" in data:
                                data["is_active"] = data.pop("active")
                            if "styles" not in data:
                                data["styles"] = []
                            if "job_id" not in data:
                                data["job_id"] = ""
                            if "dataset_id" not in data:
                                data["dataset_id"] = ""
                            
                            # Calculate size dynamically
                            total_size = 0
                            for p in ckpt_dir.rglob('*'):
                                if p.is_file():
                                    total_size += p.stat().st_size
                            data["size_bytes"] = total_size
                            
                            checkpoints.append(Checkpoint(**data))
                    except Exception as e:
                        logger.error(f"Failed to load checkpoint metadata {meta_path}: {e}")

        return checkpoints

    def get_active_checkpoint(self) -> Optional[Checkpoint]:
        """Get the currently active checkpoint."""
        for ckpt in self.list_checkpoints():
            if ckpt.is_active:
                return ckpt
        return None
    
    def activate_checkpoint(self, checkpoint_id: str) -> bool:
        """Set a checkpoint as active."""
        checkpoints = self.list_checkpoints()
        
        for ckpt in checkpoints:
            ckpt.is_active = (ckpt.id == checkpoint_id)
            meta_path = self.checkpoints_dir / ckpt.id / "meta.json"
            with open(meta_path, 'w') as f:
                json.dump(ckpt.to_dict(), f, indent=2)
        
        return any(ckpt.id == checkpoint_id for ckpt in checkpoints)
    
    def deactivate_all_checkpoints(self):
        """Deactivate all checkpoints."""
        checkpoints = self.list_checkpoints()
        for ckpt in checkpoints:
            if ckpt.is_active:
                ckpt.is_active = False
                meta_path = self.checkpoints_dir / ckpt.id / "meta.json"
                with open(meta_path, 'w') as f:
                    json.dump(ckpt.to_dict(), f, indent=2)
    
    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """Delete a checkpoint."""
        import shutil
        ckpt_dir = self.checkpoints_dir / checkpoint_id
        if ckpt_dir.exists():
            shutil.rmtree(ckpt_dir)
            logger.info(f"Deleted checkpoint {checkpoint_id}")
            return True
        return False


# Singleton instance
fine_tuning_service = FineTuningService()
