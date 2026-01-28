#!/usr/bin/env python3
"""
HeartMuLa Training Runner Script

Standalone script that runs training as a subprocess.
Writes progress to stdout for parent process to monitor.
"""

import sys
import json
import argparse
import logging
from pathlib import Path

# Setup logging to stderr (stdout reserved for progress JSON)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)


def emit_progress(epoch: int, step: int, loss: float, status: str = "running"):
    """Emit progress as JSON to stdout for parent process."""
    progress = {
        "epoch": epoch,
        "step": step,
        "loss": loss,
        "status": status
    }
    print(json.dumps(progress), flush=True)


def main():
    parser = argparse.ArgumentParser(description="HeartMuLa Training Runner")
    parser.add_argument("--job-id", required=True, help="Training job ID")
    parser.add_argument("--dataset-dir", required=True, help="Path to dataset directory")
    parser.add_argument("--output-dir", required=True, help="Path to save checkpoints")
    parser.add_argument("--model-path", required=True, help="Path to HeartMuLa model")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=float, default=16.0, help="LoRA alpha")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--method", choices=["lora", "full"], default="lora")
    parser.add_argument("--device", default="cpu", help="Device (cpu/cuda/mps)")
    
    args = parser.parse_args()
    
    logger.info(f"Starting training job: {args.job_id}")
    logger.info(f"Dataset: {args.dataset_dir}")
    logger.info(f"Output: {args.output_dir}")
    
    try:
        # Import training modules
        from app.services.training.data_prep import DataPreparator, TrainingDataset
        from app.services.training.lora_trainer import (
            HeartMuLaLoRATrainer, TrainingConfig, LoRAConfig
        )
        
        dataset_dir = Path(args.dataset_dir)
        output_dir = Path(args.output_dir)
        
        # Step 1: Prepare data
        def prep_callback(current: int, total: int, filename: str):
            percent = int((current / total) * 100)
            # Emit "preprocessing" status with percentage and message
            progress = {
                "epoch": 0,
                "step": current,
                "loss": 0.0,
                "status": "preprocessing",
                "percent": percent,
                "message": f"Processing {filename} ({current}/{total})"
            }
            print(json.dumps(progress), flush=True)

        logger.info("Preprocessing dataset...")
        
        preparator = DataPreparator(
            heartcodec_path=args.model_path,
            device=args.device
        )
        
        # Prepare samples
        prepared_dir = dataset_dir / "prepared"
        sample_paths = preparator.prepare_dataset(
            audio_dir=dataset_dir / "audio",
            manifest_path=dataset_dir / "manifest.json",
            output_dir=prepared_dir,
            progress_callback=prep_callback
        )
        
        if not sample_paths:
            emit_progress(0, 0, 0.0, "failed")
            logger.error("No samples prepared!")
            sys.exit(1)
        
        logger.info(f"Prepared {len(sample_paths)} samples")
        
        # Emit Loading Model state (still preprocessing 100%)
        print(json.dumps({
            "epoch": 0,
            "step": 0,
            "loss": 0.0,
            "status": "preprocessing",
            "percent": 100,
            "message": "Loading 3B Parameter Model..."
        }), flush=True)

        # Step 2: Create training dataset
        train_dataset = TrainingDataset(sample_paths)
        
        # Step 3: Setup trainer
        lora_config = LoRAConfig(
            rank=args.lora_rank,
            alpha=args.lora_alpha
        )
        
        training_config = TrainingConfig(
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            lora=lora_config
        )
        
        def progress_callback(epoch: int, step: int, loss: float, total_steps: int):
            percent = int((step / total_steps) * 100) if total_steps > 0 else 0
            # Ensure we don't exceed 100 or backward
            percent = min(max(percent, 0), 99)
            
            progress = {
                "epoch": epoch + 1,
                "step": step,
                "loss": loss,
                "status": "running",
                "percent": percent
            }
            print(json.dumps(progress), flush=True)
        
        def log_callback(message: str):
            logger.info(message)
        
        # This is where the model is loaded (slow)
        trainer = HeartMuLaLoRATrainer(
            model_path=args.model_path,
            config=training_config,
            device=args.device,
            log_callback=log_callback
        )
        
        # Emit RUNNING only after model load is done
        emit_progress(0, 0, 0.0, "running")
        
        # Step 4: Train!
        trainer.train(
            train_dataset=train_dataset,
            output_dir=output_dir / "checkpoints",
            progress_callback=progress_callback
        )
        
        emit_progress(args.epochs, 0, 0.0, "completed")
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.exception(f"Training failed: {e}")
        # Emit error with details for frontend
        error_progress = {
            "epoch": 0,
            "step": 0,
            "loss": 0.0,
            "status": "failed",
            "error": str(e)
        }
        print(json.dumps(error_progress), flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
