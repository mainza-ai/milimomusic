"""
API integration tests for style and training endpoints.
Tests HTTP endpoints using FastAPI TestClient.
"""
import pytest
from fastapi.testclient import TestClient
import tempfile
import os
import shutil
from pathlib import Path
import json

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.main import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


class TestStyleEndpoints:
    """Test suite for style API endpoints."""

    def test_get_styles(self, client):
        """GET /styles should return list of styles."""
        response = client.get("/styles")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0

    def test_styles_have_required_fields(self, client):
        """Each style should have name and type."""
        response = client.get("/styles")
        data = response.json()
        
        for style in data:
            assert "name" in style
            assert "type" in style
            assert style["type"] in ["official", "custom", "trained"]

    def test_add_custom_style(self, client):
        """POST /styles/custom should add a new custom style."""
        unique_name = f"TestStyle_{os.urandom(4).hex()}"
        
        response = client.post(
            "/styles/custom",
            json={"name": unique_name, "description": "Test description"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == unique_name
        assert data["type"] == "custom"
        
        # Cleanup
        client.delete(f"/styles/custom/{unique_name}")

    def test_add_duplicate_style_fails(self, client):
        """Adding duplicate style should return 400."""
        unique_name = f"DupeTest_{os.urandom(4).hex()}"
        
        # First add
        client.post("/styles/custom", json={"name": unique_name})
        
        # Second add should fail
        response = client.post("/styles/custom", json={"name": unique_name})
        assert response.status_code == 400
        
        # Cleanup
        client.delete(f"/styles/custom/{unique_name}")

    def test_delete_custom_style(self, client):
        """DELETE /styles/custom/{name} should remove style."""
        unique_name = f"ToDelete_{os.urandom(4).hex()}"
        
        # Create first
        client.post("/styles/custom", json={"name": unique_name})
        
        # Delete
        response = client.delete(f"/styles/custom/{unique_name}")
        assert response.status_code == 200
        
        # Verify deleted
        styles = client.get("/styles").json()
        names = [s["name"] for s in styles]
        assert unique_name not in names


class TestConfigEndpoints:
    """Test suite for configuration endpoints."""

    def test_get_paths(self, client):
        """GET /config/paths should return path config."""
        response = client.get("/config/paths")
        
        assert response.status_code == 200
        data = response.json()
        assert "custom_models_dir" in data

    def test_update_paths(self, client):
        """POST /config/paths should update path config."""
        original = client.get("/config/paths").json()
        
        response = client.post(
            "/config/paths",
            json={"custom_models_dir": "/test/path"}
        )
        
        assert response.status_code == 200
        
        # Restore original
        client.post("/config/paths", json=original)

    def test_validate_paths(self, client):
        """POST /config/paths/validate should check path validity."""
        response = client.post(
            "/config/paths/validate",
            json={"path": "/nonexistent/path/12345"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "valid" in data


class TestTrainingEndpoints:
    """Test suite for training API endpoints."""

    def test_list_datasets(self, client):
        """GET /training/datasets should return list."""
        response = client.get("/training/datasets")
        
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_create_dataset(self, client):
        """POST /training/datasets should create new dataset."""
        unique_name = f"TestDataset_{os.urandom(4).hex()}"
        
        response = client.post(
            "/training/datasets",
            json={"name": unique_name, "styles": ["Pop", "Rock"]}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == unique_name
        assert "id" in data

    def test_list_jobs(self, client):
        """GET /training/jobs should return list."""
        response = client.get("/training/jobs")
        
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_list_checkpoints(self, client):
        """GET /training/checkpoints should return list."""
        response = client.get("/training/checkpoints")
        
        assert response.status_code == 200
        assert isinstance(response.json(), list)


class TestHealthCheck:
    """Test basic app health."""

    def test_app_responds(self, client):
        """App should respond to requests."""
        # Try the styles endpoint as a health check
        response = client.get("/styles")
        assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
