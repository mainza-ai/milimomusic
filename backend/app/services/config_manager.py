import json
import os
from typing import Dict, Any, Optional
import logging

from dotenv import load_dotenv

# Load environment variables from ../.env (repo root) so secrets live outside source.
# ConfigManager may be imported from backend/ cwd; search upward for the repo root.
_ENV_LOADED = False


def _load_dotenv_once():
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    _ENV_LOADED = True
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env"))
    # Also allow a backend/.env
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))


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
    "opencode": {
        "api_key": "",
        "base_url": "https://opencode.ai/zen/go/v1",
        "model": "minimax-m3"
    },
    "omlx": {
        "base_url": "http://localhost:8787/v1",
        "api_key": "omlx",
        "model": "Llama-3.2-3B-Instruct-bf16"
    },
    "paths": {
        "model_directory": "../heartlib/ckpt",
        "checkpoints_directory": "./data/checkpoints",
        "datasets_directory": "./data/datasets"
    }
}

# map: config_key -> (env_var, default)
_ENV_MAP = {
    "opencode": {"api_key": ("OPENCODE_API_KEY", ""), "base_url": ("OPENCODE_BASE_URL", "https://opencode.ai/zen/go/v1"), "model": ("OPENCODE_MODEL", "minimax-m3")},
    "deepseek": {"api_key": ("DEEPSEEK_API_KEY", ""), "base_url": ("DEEPSEEK_BASE_URL", "https://api.deepseek.com"), "model": ("DEEPSEEK_MODEL", "deepseek-chat")},
    "openrouter": {"api_key": ("OPENROUTER_API_KEY", ""), "base_url": ("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"), "model": ("OPENROUTER_MODEL", "openai/gpt-3.5-turbo")},
    "openai": {"api_key": ("OPENAI_API_KEY", ""), "model": ("OPENAI_MODEL", "gpt-4o")},
    "gemini": {"api_key": ("GEMINI_API_KEY", ""), "model": ("GEMINI_MODEL", "gemini-1.5-flash")},
    "omlx": {"api_key": ("OMLX_API_KEY", "omlx"), "base_url": ("OMLX_BASE_URL", "http://localhost:8787/v1"), "model": ("OMLX_MODEL", "Llama-3.2-3B-Instruct-bf16")},
    "ollama": {"base_url": ("OLLAMA_BASE_URL", "http://localhost:11434"), "model": ("OLLAMA_MODEL", "llama3.2:3b-instruct-fp16")},
    "lmstudio": {"api_key": ("LMSTUDIO_API_KEY", "lm-studio"), "base_url": ("LMSTUDIO_BASE_URL", "http://localhost:1234/v1"), "model": ("LMSTUDIO_MODEL", "local-model")},
}


def _apply_env_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    """Overlay .env / os.environ values on top of the config dict (no secrets hardcoded).
    A non-empty env value wins; an empty/absent env value falls back to the file/default."""
    _load_dotenv_once()
    for provider, fields in _ENV_MAP.items():
        provider_cfg = config.setdefault(provider, {})
        for field, (env_var, default) in fields.items():
            env_val = os.environ.get(env_var)
            if env_val:
                provider_cfg[field] = env_val
            else:
                provider_cfg[field] = provider_cfg.get(field, default)
    # Paths
    paths = config.setdefault("paths", {})
    for field, env_var, fallback in (
        ("model_directory", "MODEL_DIRECTORY", "../heartlib/ckpt"),
        ("checkpoints_directory", "CHECKPOINTS_DIRECTORY", "./data/checkpoints"),
        ("datasets_directory", "DATASETS_DIRECTORY", "./data/datasets"),
    ):
        env_val = os.environ.get(env_var)
        if env_val:
            paths[field] = env_val
        else:
            paths[field] = paths.get(field, fallback)
    return config


class ConfigManager:
    _instance = None
    _config: Dict[str, Any] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self):
        """Load configuration from file or use defaults, then overlay .env."""
        _load_dotenv_once()
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
        # Environment variables take precedence over file/default (no secrets hardcoded)
        self._config = _apply_env_overrides(self._config)
        # Preserve the active provider, defaulting to the configured one
        self._config["provider"] = os.environ.get("LLM_PROVIDER", self._config.get("provider", "ollama"))

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
