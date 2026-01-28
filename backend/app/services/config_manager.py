import json
import os
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

CONFIG_FILE = "llm_config.json"

DEFAULT_CONFIG = {
    "provider": "ollama",
    "ollama": {
        "base_url": "http://localhost:11434",
        "model": "llama3.2:3b-instruct-fp16"
    },
    "openai": {
        "api_key": "",
        "model": "gpt-4o"
    },
    "gemini": {
        "api_key": "",
        "model": "gemini-1.5-flash"
    },
    "openrouter": {
        "api_key": "",
        "model": "openai/gpt-3.5-turbo"
    },
    "lmstudio": {
        "base_url": "http://localhost:1234/v1",
        "model": "local-model"
    },
    "deepseek": {
        "api_key": "",
        "model": "deepseek-chat"
    },
    "paths": {
        "model_directory": "../heartlib/ckpt",
        "checkpoints_directory": "./data/checkpoints",
        "datasets_directory": "./data/datasets"
    }
}

class ConfigManager:
    _instance = None
    _config: Dict[str, Any] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self):
        """Load configuration from file or use defaults."""
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r') as f:
                    loaded_config = json.load(f)
                    # Merge with default config to ensure all keys exist
                    self._config = self._merge_configs(DEFAULT_CONFIG, loaded_config)
            except Exception as e:
                logger.error(f"Failed to load config file: {e}")
                self._config = DEFAULT_CONFIG.copy()
        else:
            self._config = DEFAULT_CONFIG.copy()
            self.save_config()

    def _merge_configs(self, default: Dict, loaded: Dict) -> Dict:
        """Deep merge default and loaded configs."""
        result = default.copy()
        for key, value in loaded.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        return result

    def save_config(self):
        """Save current configuration to file."""
        try:
            with open(CONFIG_FILE, 'w') as f:
                json.dump(self._config, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to save config file: {e}")

    def get_config(self) -> Dict[str, Any]:
        """Get the entire configuration."""
        return self._config

    def get_provider_config(self, provider_name: str) -> Dict[str, Any]:
        """Get configuration for a specific provider."""
        return self._config.get(provider_name, {})

    def update_config(self, new_config: Dict[str, Any]):
        """Update configuration with partial data."""
        self._config = self._merge_configs(self._config, new_config)
        self.save_config()

    def set_provider(self, provider_name: str):
        """Set the active provider."""
        if provider_name in self._config:
            self._config["provider"] = provider_name
            self.save_config()
        else:
            raise ValueError(f"Unknown provider: {provider_name}")
