"""
Unit tests for StyleRegistry service.
Tests style loading, custom style management, and trained style registration.
"""
import pytest
import os
from pathlib import Path

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.style_registry import StyleRegistry, OFFICIAL_STYLES, Style


class TestStyleRegistry:
    """Test suite for StyleRegistry service."""

    @pytest.fixture
    def registry(self):
        """Get the StyleRegistry singleton."""
        return StyleRegistry()

    def test_get_all_styles_includes_official(self, registry):
        """Official styles should always be present."""
        styles = registry.get_all_styles()
        style_names = [s.name for s in styles]
        
        # Check some known official styles exist
        assert 'Pop' in style_names
        assert 'Rock' in style_names
        assert 'Electronic' in style_names

    def test_get_all_styles_returns_style_objects(self, registry):
        """All styles should be Style objects with valid types."""
        styles = registry.get_all_styles()
        
        valid_types = {'official', 'custom', 'trained'}
        for style in styles:
            assert isinstance(style, Style)
            assert style.type in valid_types

    def test_get_official_styles(self, registry):
        """Official styles should match OFFICIAL_STYLES constant."""
        official = registry.get_official_styles()
        
        assert len(official) == len(OFFICIAL_STYLES)
        for style in official:
            assert style.type == 'official'
            assert style.name in OFFICIAL_STYLES

    def test_add_custom_style(self, registry):
        """Adding a custom style should work."""
        unique_name = f"TestStyle_{os.urandom(4).hex()}"
        
        try:
            style = registry.add_custom_style(unique_name, "A test style")
            
            assert style.name == unique_name
            assert style.type == 'custom'
            assert style.description == 'A test style'
            
            # Verify it's in the list
            all_styles = registry.get_all_styles()
            names = [s.name for s in all_styles]
            assert unique_name in names
        finally:
            # Cleanup
            registry.remove_custom_style(unique_name)

    def test_add_duplicate_official_style_fails(self, registry):
        """Adding a style with same name as official should fail."""
        with pytest.raises(ValueError) as exc:
            registry.add_custom_style("Pop")
        
        assert "official" in str(exc.value).lower()

    def test_add_duplicate_custom_style_fails(self, registry):
        """Adding duplicate custom style should fail."""
        unique_name = f"DupeTest_{os.urandom(4).hex()}"
        
        try:
            registry.add_custom_style(unique_name)
            
            with pytest.raises(ValueError) as exc:
                registry.add_custom_style(unique_name)
            
            assert "already exists" in str(exc.value)
        finally:
            registry.remove_custom_style(unique_name)

    def test_remove_custom_style(self, registry):
        """Removing a custom style should work."""
        unique_name = f"ToRemove_{os.urandom(4).hex()}"
        
        registry.add_custom_style(unique_name)
        result = registry.remove_custom_style(unique_name)
        
        assert result is True
        
        # Verify it's gone
        all_styles = registry.get_all_styles()
        names = [s.name for s in all_styles]
        assert unique_name not in names

    def test_remove_nonexistent_style_returns_false(self, registry):
        """Removing nonexistent style should return False."""
        result = registry.remove_custom_style("NonexistentStyle12345XYZ")
        assert result is False

    def test_get_custom_styles(self, registry):
        """get_custom_styles should return only custom styles."""
        unique_name = f"CustomOnly_{os.urandom(4).hex()}"
        
        try:
            registry.add_custom_style(unique_name)
            
            custom = registry.get_custom_styles()
            
            # All should be custom or trained type
            for style in custom:
                assert style.type in ('custom', 'trained')
            
            # Our style should be there
            names = [s.name for s in custom]
            assert unique_name in names
        finally:
            registry.remove_custom_style(unique_name)

    def test_get_styles_for_prompt(self, registry):
        """get_styles_for_prompt should return list of names."""
        result = registry.get_styles_for_prompt()
        
        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(s, str) for s in result)
        assert 'Pop' in result

    def test_promote_to_trained(self, registry):
        """Promoting style to trained should update type."""
        unique_name = f"ToTrain_{os.urandom(4).hex()}"
        
        try:
            registry.add_custom_style(unique_name)
            
            result = registry.promote_to_trained(unique_name, "checkpoint-123")
            
            assert result is not None
            assert result.type == 'trained'
            assert result.checkpoint_id == 'checkpoint-123'
        finally:
            registry.remove_custom_style(unique_name)

    def test_promote_nonexistent_returns_none(self, registry):
        """Promoting nonexistent style should return None."""
        result = registry.promote_to_trained("NonexistentXYZ", "ckpt-1")
        assert result is None

    def test_singleton_pattern(self):
        """StyleRegistry should be a singleton."""
        reg1 = StyleRegistry()
        reg2 = StyleRegistry()
        
        assert reg1 is reg2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
