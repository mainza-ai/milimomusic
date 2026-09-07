import copy
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
    # Search root .env and backend/.env
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    load_dotenv(os.path.join(repo_root, ".env"), override=True)
    load_dotenv(os.path.join(repo_root, "backend", ".env"), override=True)
    load_dotenv(".env", override=True)


logger = logging.getLogger(__name__)

CONFIG_FILE = "llm_config.json"

DEFAULT_CONFIG = {
    "provider": "opencode",
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
    "nvidia": {
        "api_key": "",
        "base_url": "https://integrate.api.nvidia.com/v1",
        "model": "deepseek-ai/deepseek-v4-flash-0731"
    },
    "deepseek": {
        "api_key": "",
        "model": "deepseek-chat"
    },
    "opencode": {
        "api_key": "",
        "base_url": "https://opencode.ai/zen/go/v1",
        "model": "deepseek-v4-flash"
    },
    "omlx": {
        "base_url": "http://localhost:8787/v1",
        "api_key": "omlx",
        "model": "Llama-3.2-3B-Instruct-bf16"
    },
    "paths": {
        "models_directory": "./models",
        "model_directory": "./models",
        "checkpoints_directory": "./data/checkpoints",
        "datasets_directory": "./data/datasets",
        "heartmula_model_path": "../heartlib/ckpt"
    }
}

# map: config_key -> (env_var, default)
_ENV_MAP = {
    "nvidia": {"api_key": ("NVIDIA_API_KEY", ""), "base_url": ("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"), "model": ("NVIDIA_MODEL", "deepseek-ai/deepseek-v4-flash-0731")},
    "opencode": {"api_key": ("OPENCODE_API_KEY", ""), "base_url": ("OPENCODE_BASE_URL", "https://opencode.ai/zen/go/v1"), "model": ("OPENCODE_MODEL", "deepseek-v4-flash")},
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
    # Active provider from env if provided
    env_provider = os.environ.get("LLM_PROVIDER")
    if env_provider:
        config["provider"] = env_provider
    # Paths
    paths = config.setdefault("paths", {})

    # Models directory: check MODELS_DIRECTORY or MODEL_DIRECTORY (if not legacy heartlib/ckpt)
    env_models = os.environ.get("MODELS_DIRECTORY") or os.environ.get("MODEL_DIRECTORY")
    if env_models and not env_models.strip().endswith("heartlib/ckpt") and not env_models.strip().endswith("heartlib/ckpt/"):
        paths["models_directory"] = env_models
        paths["model_directory"] = env_models
    else:
        current_md = paths.get("models_directory") or paths.get("model_directory")
        if not current_md or current_md.endswith("heartlib/ckpt") or current_md.endswith("heartlib/ckpt/"):
            paths["models_directory"] = "./models"
            paths["model_directory"] = "./models"
        else:
            paths["models_directory"] = current_md
            paths["model_directory"] = current_md

    # HeartMuLa legacy path
    env_hm = os.environ.get("HEARTMULA_MODEL_PATH")
    if env_hm:
        paths["heartmula_model_path"] = env_hm
    else:
        paths.setdefault("heartmula_model_path", "../heartlib/ckpt")

    for field, env_var, fallback in (
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
    _file_config: Dict[str, Any] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self):
        """Load configuration from file or use defaults without polluting with env secrets."""
        _load_dotenv_once()
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r') as f:
                    loaded_config = json.load(f)
                    self._file_config = self._merge_configs(DEFAULT_CONFIG, loaded_config)
            except Exception as e:
                logger.error(f"Failed to load config file: {e}")
                self._file_config = copy.deepcopy(DEFAULT_CONFIG)
        else:
            self._file_config = copy.deepcopy(DEFAULT_CONFIG)

    def _merge_configs(self, default: Dict, loaded: Dict) -> Dict:
        """Deep merge default and loaded configs, protecting existing api_keys from blank overwrites."""
        result = copy.deepcopy(default)
        for key, value in loaded.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                for sub_key, sub_val in value.items():
                    if sub_key == "api_key" and (sub_val is None or str(sub_val).strip() == ""):
                        continue
                    result[key][sub_key] = sub_val
            else:
                result[key] = value
        return result

    def save_config(self):
        """Save current user file configuration without dumping environment secrets."""
        try:
            to_save = copy.deepcopy(self._file_config)
            # Never persist keys that match environment variables to llm_config.json
            for provider, fields in _ENV_MAP.items():
                if provider in to_save and isinstance(to_save[provider], dict):
                    for field, (env_var, _) in fields.items():
                        env_val = os.environ.get(env_var)
                        if env_val and to_save[provider].get(field) == env_val:
                            if field == "api_key":
                                to_save[provider][field] = ""
            with open(CONFIG_FILE, 'w') as f:
                json.dump(to_save, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to save config file: {e}")

    def get_config(self) -> Dict[str, Any]:
        """Get the full runtime configuration including .env secrets (for internal backend services)."""
        runtime_config = copy.deepcopy(self._file_config)
        return _apply_env_overrides(runtime_config)

    def get_client_config(self) -> Dict[str, Any]:
        """Get sanitized configuration for client consumption (masks api_keys and returns has_key booleans)."""
        runtime_config = self.get_config()
        client_cfg = copy.deepcopy(runtime_config)
        for provider, fields in _ENV_MAP.items():
            if provider in client_cfg and isinstance(client_cfg[provider], dict):
                api_key = client_cfg[provider].get("api_key", "")
                has_key = bool(api_key and str(api_key).strip() and api_key != "omlx" and api_key != "lm-studio")
                if provider in ("omlx", "lmstudio"):
                    has_key = True
                client_cfg[provider]["has_key"] = has_key
                client_cfg[provider]["has_api_key"] = has_key
                # Strip plaintext secret from client payload
                client_cfg[provider]["api_key"] = ""
        return client_cfg

    def get_provider_config(self, provider_name: str) -> Dict[str, Any]:
        """Get runtime configuration for a specific provider."""
        return self.get_config().get(provider_name, {})

    def update_config(self, new_config: Dict[str, Any]):
        """Update configuration with user data and persist safe non-secret fields."""
        self._file_config = self._merge_configs(self._file_config, new_config)
        self.save_config()

    def set_provider(self, provider_name: str):
        """Set the active provider."""
        if provider_name in DEFAULT_CONFIG or provider_name in self._file_config:
            self._file_config["provider"] = provider_name
            self.save_config()
        else:
            raise ValueError(f"Unknown provider: {provider_name}")

