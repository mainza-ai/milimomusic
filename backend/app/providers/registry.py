"""
Generation Provider Registry and Capability Negotiation.
"""

import logging
from typing import Dict, List, Optional
from app.providers.base import GenerationProvider, GenerationCapabilities
from app.providers.minimax_provider import MiniMaxMusic3Provider

logger = logging.getLogger(__name__)


class ProviderRegistry:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ProviderRegistry, cls).__new__(cls)
            cls._instance.providers: Dict[str, GenerationProvider] = {}
            cls._instance.active_provider_id: str = "minimax_music3"
            cls._instance._register_defaults()
        return cls._instance

    def _register_defaults(self):
        """Register MiniMax Music 3 as primary generation engine."""
        minimax = MiniMaxMusic3Provider()
        self.register_provider("minimax_music3", minimax)

    def register_provider(self, provider_id: str, provider: GenerationProvider):
        self.providers[provider_id] = provider
        if "/" in provider_id and not provider_id.startswith("hf:"):
            self.providers[f"hf:{provider_id}"] = provider
        logger.info(f"Registered generation provider: {provider_id}")

    @classmethod
    def get_provider(cls, provider_id: Optional[str] = None) -> GenerationProvider:
        inst = cls()
        return inst._get_provider_impl(provider_id)

    def _get_provider_impl(self, provider_id: Optional[str] = None) -> GenerationProvider:
        target_id = provider_id or self.active_provider_id
        if target_id in self.providers:
            return self.providers[target_id]

        cleaned_id = target_id.removeprefix("hf:")
        if cleaned_id in self.providers:
            return self.providers[cleaned_id]

        # Check if this is a Hugging Face model repository or custom downloaded model
        from app.providers.hf_audio_provider import HuggingFaceAudioProvider
        if "/" in cleaned_id:
            provider = HuggingFaceAudioProvider(cleaned_id)
            self.register_provider(target_id, provider)
            self.register_provider(cleaned_id, provider)
            return provider

        # Check if custom model registered in model_manager
        try:
            from app.services.model_manager import model_manager
            custom_models = model_manager._load_custom_models()
            match = next((m for m in custom_models if m.get("id") == target_id or m.get("repo_id") == target_id or m.get("repo_id") == cleaned_id), None)
            if match:
                repo_id = match.get("repo_id", cleaned_id)
                provider = HuggingFaceAudioProvider(repo_id, local_path=match.get("local_path"))
                self.register_provider(target_id, provider)
                self.register_provider(cleaned_id, provider)
                return provider
        except Exception:
            pass

        logger.warning(f"Provider '{target_id}' not found, falling back to 'minimax_music3'")
        fallback_id = "minimax_music3" if "minimax_music3" in self.providers else list(self.providers.keys())[0]
        return self.providers[fallback_id]

    def set_active_provider(self, provider_id: str) -> bool:
        if provider_id in self.providers:
            self.active_provider_id = provider_id
            logger.info(f"Active provider set to: {provider_id}")
            return True
        return False

    def get_active_provider_id(self) -> str:
        return self.active_provider_id

    def list_capabilities(self) -> List[GenerationCapabilities]:
        return [p.get_capabilities() for p in self.providers.values()]

    def get_active_capabilities(self) -> GenerationCapabilities:
        return self.get_provider().get_capabilities()


provider_registry = ProviderRegistry()
