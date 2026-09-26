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
        """Register default generation engines."""
        minimax = MiniMaxMusic3Provider()
        for mid in [
            "minimax_music3", "minimax", "minimax_music3_mxfp4", "minimax_music3_4bit",
            "minimax_music3_6bit", "minimax_music3_8bit", "minimax_music3_bf16",
            "minimax_music3_comfy_int8", "minimax_music3_gguf_q4", "minimax_music3_official_pytorch"
        ]:
            self.register_provider(mid, minimax)
        try:
            from app.providers.heartmula_provider import HeartMuLaProvider
            heartmula = HeartMuLaProvider()
            self.register_provider("heartmula", heartmula)
            self.register_provider("heartmula_3b", heartmula)
        except Exception as e:
            logger.warning(f"HeartMuLaProvider registration deferred: {e}")

        try:
            from app.providers.yue2_provider import YuE2Provider
            self.register_provider("yue2", YuE2Provider())
        except Exception as e:
            logger.warning(f"YuE2Provider registration deferred: {e}")

        try:
            import sys
            from pathlib import Path
            repo_root = Path(__file__).resolve().parents[3]
            mulacover_src = repo_root / "mulacover" / "src"
            if mulacover_src.exists() and str(mulacover_src) not in sys.path:
                sys.path.insert(0, str(mulacover_src))
            from app.providers.mulacover_provider import mulacover_provider
            self.register_provider("mulacover", mulacover_provider)
        except Exception as e:
            logger.warning(f"MuLaCoverProvider registration deferred: {e}")

        try:
            from app.providers.stable_audio_provider import StableAudioProvider
            stable_audio = StableAudioProvider()
            self.register_provider("stable_audio_open_1_0", stable_audio)
            self.register_provider("stable_audio", stable_audio)
            self.register_provider("stabilityai/stable-audio-open-1.0", stable_audio)
        except Exception as e:
            logger.warning(f"StableAudioProvider registration deferred: {e}")

        try:
            from app.providers.musicgen_provider import MusicGenProvider
            musicgen_small = MusicGenProvider("facebook/musicgen-small")
            musicgen_melody = MusicGenProvider("facebook/musicgen-melody")
            self.register_provider("musicgen_small", musicgen_small)
            self.register_provider("musicgen", musicgen_small)
            self.register_provider("facebook/musicgen-small", musicgen_small)
            self.register_provider("musicgen_melody", musicgen_melody)
            self.register_provider("facebook/musicgen-melody", musicgen_melody)
        except Exception as e:
            logger.warning(f"MusicGenProvider registration deferred: {e}")

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

        # Check if stable-audio or musicgen
        if "stable-audio" in cleaned_id or "stable_audio" in cleaned_id:
            from app.providers.stable_audio_provider import StableAudioProvider
            provider = StableAudioProvider(cleaned_id if "/" in cleaned_id else None)
            self.register_provider(target_id, provider)
            self.register_provider(cleaned_id, provider)
            return provider

        if "musicgen" in cleaned_id:
            from app.providers.musicgen_provider import MusicGenProvider
            repo = cleaned_id if "/" in cleaned_id else ("facebook/musicgen-melody" if "melody" in cleaned_id else "facebook/musicgen-small")
            provider = MusicGenProvider(repo)
            self.register_provider(target_id, provider)
            self.register_provider(cleaned_id, provider)
            return provider

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
        cleaned = provider_id.removeprefix("hf:")
        if cleaned in self.providers:
            self.active_provider_id = cleaned
            return True
        if "minimax" in provider_id.lower() and "minimax_music3" in self.providers:
            self.active_provider_id = "minimax_music3"
            return True
        return False

    def get_active_provider_id(self) -> str:
        return self.active_provider_id

    def list_capabilities(self) -> List[GenerationCapabilities]:
        return [p.get_capabilities() for p in self.providers.values()]

    def get_active_capabilities(self) -> GenerationCapabilities:
        return self.get_provider().get_capabilities()

    def unload_all(self) -> int:
        """Unload all instantiated providers from memory."""
        count = 0
        for p in set(self.providers.values()):
            try:
                if hasattr(p, "unload"):
                    p.unload()
                    count += 1
            except Exception as e:
                logger.warning(f"Error unloading provider {p}: {e}")
        return count


provider_registry = ProviderRegistry()

try:
    from app.core.hardware_lock import GlobalHardwareCoordinator
    GlobalHardwareCoordinator.register_eviction_hook(
        "audio_gen",
        lambda: provider_registry.unload_all()
    )
except Exception:
    pass


def get_provider(provider_id: Optional[str] = None) -> GenerationProvider:
    return ProviderRegistry.get_provider(provider_id)


def list_providers() -> List[str]:
    return list(provider_registry.providers.keys())
