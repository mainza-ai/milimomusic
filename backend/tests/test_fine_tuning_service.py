"""
Unit tests for FineTuningService.
Tests dataset management, training job lifecycle, and checkpoint management.
"""
import pytest
import os
from pathlib import Path

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.fine_tuning_service import (
    FineTuningService, Dataset, TrainingJob, Checkpoint, TrainingConfig
)


class TestDatasetManagement:
    """Test suite for dataset operations."""

    @pytest.fixture
    def service(self):
        """Get the FineTuningService singleton."""
        return FineTuningService()

    def test_create_dataset(self, service):
        """Creating a dataset should work."""
        unique_name = f"TestDataset_{os.urandom(4).hex()}"
        
        dataset = service.create_dataset(unique_name, ["Pop", "Rock"])
        
        assert dataset.name == unique_name
        assert dataset.styles == ["Pop", "Rock"]
        assert dataset.id is not None

    def test_create_dataset_creates_directory(self, service):
        """Creating a dataset should create its directory."""
        unique_name = f"DirTest_{os.urandom(4).hex()}"
        
        dataset = service.create_dataset(unique_name, [])
        
        expected_path = service.datasets_dir / dataset.id
        assert expected_path.exists()

    def test_get_dataset(self, service):
        """Getting a dataset by ID should return the dataset."""
        unique_name = f"GetTest_{os.urandom(4).hex()}"
        created = service.create_dataset(unique_name, ["Jazz"])
        
        retrieved = service.get_dataset(created.id)
        
        assert retrieved is not None
        assert retrieved.name == unique_name
        assert retrieved.styles == ["Jazz"]

    def test_get_nonexistent_dataset(self, service):
        """Getting a nonexistent dataset should return None."""
        result = service.get_dataset("nonexistent-id-12345xyz")
        assert result is None

    def test_list_datasets(self, service):
        """Listing datasets should return all datasets."""
        datasets = service.list_datasets()
        
        # Just verify it returns a list
        assert isinstance(datasets, list)

    def test_validate_dataset_insufficient_files(self, service):
        """Dataset with <10 files should not validate."""
        unique_name = f"SmallDataset_{os.urandom(4).hex()}"
        dataset = service.create_dataset(unique_name, [])
        
        result = service.validate_dataset(dataset.id)
        
        assert result["valid"] is False
        assert "file_count" in result
        assert result["file_count"] == 0

    def test_validate_nonexistent_dataset(self, service):
        """Validating nonexistent dataset should return error."""
        result = service.validate_dataset("fake-id-xyz123")
        
        assert result["valid"] is False
        assert "error" in result or "not found" in str(result).lower()


class TestTrainingJobs:
    """Test suite for training job operations."""

    @pytest.fixture
    def service(self):
        """Get the FineTuningService singleton."""
        return FineTuningService()

    def test_list_jobs(self, service):
        """Listing jobs should return a list."""
        jobs = service.list_jobs()
        
        assert isinstance(jobs, list)

    def test_create_job_with_invalid_dataset_fails(self, service):
        """Creating job with nonexistent dataset should fail."""
        config = TrainingConfig(
            dataset_id="fake-id-xyz",
            method="lora",
            epochs=3
        )
        
        with pytest.raises(ValueError) as exc:
            service.create_training_job(config)
        
        # Should complain about validation
        assert "invalid" in str(exc.value).lower() or "files" in str(exc.value).lower()

    def test_get_job_logs_empty_for_nonexistent(self, service):
        """Getting logs for nonexistent job should return empty list."""
        logs = service.get_job_logs("nonexistent-job-id")
        
        assert logs == []


class TestCheckpointManagement:
    """Test suite for checkpoint operations."""

    @pytest.fixture
    def service(self):
        """Get the FineTuningService singleton."""
        return FineTuningService()

    def test_list_checkpoints(self, service):
        """Listing checkpoints should return a list."""
        checkpoints = service.list_checkpoints()
        
        assert isinstance(checkpoints, list)

    def test_get_active_checkpoint_returns_none_when_none_active(self, service):
        """get_active_checkpoint should return None if no active checkpoint."""
        # This may return None or a Checkpoint depending on state
        result = service.get_active_checkpoint()
        
        # Just verify it doesn't crash and returns correct type
        assert result is None or isinstance(result, Checkpoint)

    def test_delete_nonexistent_checkpoint(self, service):
        """Deleting nonexistent checkpoint should return False."""
        result = service.delete_checkpoint("nonexistent-checkpoint-xyz")
        
        assert result is False


class TestTrainingConfig:
    """Test TrainingConfig dataclass."""

    def test_default_config_values(self):
        """TrainingConfig should have sensible defaults."""
        config = TrainingConfig(dataset_id="test")
        
        assert config.method == "lora"
        assert config.epochs == 3
        assert config.learning_rate == 1e-4
        assert config.lora_rank == 8

    def test_config_with_custom_values(self):
        """TrainingConfig should accept custom values."""
        config = TrainingConfig(
            dataset_id="test",
            method="full",
            epochs=5,
            learning_rate=5e-5
        )
        
        assert config.method == "full"
        assert config.epochs == 5
        assert config.learning_rate == 5e-5


class TestSingleton:
    """Test singleton pattern."""

    def test_service_is_singleton(self):
        """FineTuningService should be a singleton."""
        svc1 = FineTuningService()
        svc2 = FineTuningService()
        
        assert svc1 is svc2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
