"""
StyleRegistry - Dynamic style/tag management for HeartMuLa.

Manages official HeartMuLa tags and user-defined custom styles.
Custom styles are persisted to ~/.milimo/custom_styles.json.
"""

import json
import os
from typing import List, Optional, Literal
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Official HeartMuLa tags - these are baked into the model
OFFICIAL_STYLES = [
    "Warm", "Reflection", "Pop", "Cafe", "R&B", "Keyboard", "Regret", "Drum machine",
    "Electric guitar", "Synthesizer", "Soft", "Energetic", "Electronic", "Self-discovery",
    "Sad", "Ballad", "Longing", "Meditation", "Faith", "Acoustic", "Peaceful", "Wedding",
    "Piano", "Strings", "Acoustic guitar", "Romantic", "Drums", "Emotional", "Walking",
    "Hope", "Hopeful", "Powerful", "Epic", "Driving", "Rock"
]


@dataclass
class Style:
    """Represents a musical style/tag."""
    name: str
    type: Literal["official", "custom", "trained"]
    description: Optional[str] = None
    checkpoint_id: Optional[str] = None  # For trained styles

    def to_dict(self) -> dict:
        return asdict(self)


class StyleRegistry:
    """
    Manages official and custom styles.
    
    Official styles are read-only (baked into HeartMuLa).
    Custom styles can be added/removed by users.
    Trained styles are linked to fine-tuned checkpoints.
    """
    
    _instance = None
    _custom_styles: List[Style] = []
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(StyleRegistry, cls).__new__(cls)
            cls._instance._init_paths()
            cls._instance._load_styles()
        return cls._instance
    
    def _init_paths(self):
        from .config_manager import ConfigManager
        config = ConfigManager().get_config()
        # Default to backend/data if not set
        self._config_dir = Path(os.path.expanduser(config.get("paths", {}).get("datasets_directory", "./data/datasets"))).parent
        self._styles_file = self._config_dir / "custom_styles.json"
        self.checkpoints_dir = Path(os.path.expanduser(config.get("paths", {}).get("checkpoints_directory", "./data/checkpoints")))

    def _load_styles(self):
        """Load styles from disk and discover from checkpoints."""
        self._custom_styles = []
        
        # 1. Load manually added custom styles
        if self._styles_file.exists():
            try:
                with open(self._styles_file, 'r') as f:
                    data = json.load(f)
                    for item in data.get("styles", []):
                        self._custom_styles.append(Style(**item))
            except Exception as e:
                logger.error(f"Failed to load custom styles: {e}")
        
        # 2. Discover trained styles from checkpoints
        self._discover_trained_styles()
        
        logger.info(f"Loaded {len(self._custom_styles)} total custom/trained styles")
    
    def refresh(self):
        """
        Refresh styles by re-scanning checkpoints.
        Call this after creating/activating checkpoints.
        """
        # Clear trained styles and re-discover
        self._custom_styles = [s for s in self._custom_styles if s.type == "custom"]
        self._discover_trained_styles()
        logger.info(f"Refreshed styles - now have {len(self._custom_styles)} custom/trained styles")

    def _discover_trained_styles(self):
        """Scan checkpoints directory for trained styles."""
        if not self.checkpoints_dir.exists():
            return

        existing_names = {s.name.lower() for s in self.get_all_styles()}
        
        for ckpt_dir in self.checkpoints_dir.iterdir():
            if ckpt_dir.is_dir():
                meta_path = ckpt_dir / "meta.json"
                if meta_path.exists():
                    try:
                        with open(meta_path, 'r') as f:
                            meta = json.load(f)
                            # Look for 'styles' list in meta
                            styles = meta.get("styles", [])
                            # Also check the checkpoint name itself if no explicit styles
                            if not styles and "-" in meta.get("name", ""):
                                # Heuristic: "Afrobeat-lora" -> "Afrobeat"
                                probable_style = meta["name"].split("-")[0]
                                styles = [probable_style]
                            
                            for style_name in styles:
                                if style_name.lower() not in existing_names:
                                    # Add discovered style
                                    new_style = Style(
                                        name=style_name, 
                                        type="trained", 
                                        description=f"Trained via {meta.get('method', 'lora')}",
                                        checkpoint_id=meta.get("id")
                                    )
                                    self._custom_styles.append(new_style)
                                    existing_names.add(style_name.lower())
                                    
                    except Exception as e:
                        logger.warning(f"Failed to parse checkpoint meta {meta_path}: {e}")

    def _save_custom_styles(self):
        """Persist custom styles to disk."""
        # Only save type='custom', do not save discovered 'trained' styles to json
        # (they are re-discovered on boot)
        self._config_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Filter only "custom" type for persistence
            custom_only = [s.to_dict() for s in self._custom_styles if s.type == "custom"]
            
            data = {
                "styles": custom_only
            }
            with open(self._styles_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save custom styles: {e}")
    
    def get_official_styles(self) -> List[Style]:
        """Get all official HeartMuLa styles."""
        return [Style(name=name, type="official") for name in OFFICIAL_STYLES]
    
    def get_custom_styles(self) -> List[Style]:
        """Get all user-defined custom styles."""
        return self._custom_styles.copy()
    
    def get_trained_styles(self) -> List[Style]:
        """Get styles that have associated fine-tuned checkpoints."""
        return [s for s in self._custom_styles if s.type == "trained"]
    
    def get_all_styles(self) -> List[Style]:
        """Get all styles (official + custom + trained)."""
        return self.get_official_styles() + self._custom_styles
    
    def add_custom_style(self, name: str, description: Optional[str] = None) -> Style:
        """
        Add a new custom style.
        
        Args:
            name: Style name (e.g., "Samba", "Chiptune")
            description: Optional description
            
        Returns:
            The created Style object
            
        Raises:
            ValueError: If style already exists
        """
        # Check for duplicates (case-insensitive)
        name_lower = name.lower()
        
        for official in OFFICIAL_STYLES:
            if official.lower() == name_lower:
                raise ValueError(f"'{name}' is already an official style")
        
        for custom in self._custom_styles:
            if custom.name.lower() == name_lower:
                raise ValueError(f"Custom style '{name}' already exists")
        
        style = Style(name=name, type="custom", description=description)
        self._custom_styles.append(style)
        self._save_custom_styles()
        
        logger.info(f"Added custom style: {name}")
        return style
    
    def remove_custom_style(self, name: str) -> bool:
        """
        Remove a custom style.
        
        Args:
            name: Style name to remove
            
        Returns:
            True if removed, False if not found
        """
        name_lower = name.lower()
        
        for i, style in enumerate(self._custom_styles):
            if style.name.lower() == name_lower:
                removed = self._custom_styles.pop(i)
                self._save_custom_styles()
                logger.info(f"Removed custom style: {removed.name}")
                return True
        
        return False
    
    def promote_to_trained(self, style_name: str, checkpoint_id: str) -> Optional[Style]:
        """
        Promote a custom style to 'trained' status with checkpoint link.
        
        Args:
            style_name: Name of the style
            checkpoint_id: ID of the associated checkpoint
            
        Returns:
            Updated Style object, or None if not found
        """
        name_lower = style_name.lower()
        
        for style in self._custom_styles:
            if style.name.lower() == name_lower:
                style.type = "trained"
                style.checkpoint_id = checkpoint_id
                self._save_custom_styles()
                logger.info(f"Promoted style '{style_name}' to trained with checkpoint {checkpoint_id}")
                return style
        
        return None
    
    def get_styles_for_prompt(self) -> List[str]:
        """Get all style names as a flat list for LLM prompts."""
        return [s.name for s in self.get_all_styles()]
