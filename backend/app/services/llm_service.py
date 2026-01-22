import requests
import logging
from typing import List, Optional, Dict, Any
import json
import random
from abc import ABC, abstractmethod
import os

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None

from .config_manager import ConfigManager

logger = logging.getLogger(__name__)

VALID_HEARTMULA_TAGS = [
    "Warm", "Reflection", "Pop", "Cafe", "R&B", "Keyboard", "Regret", "Drum machine",
    "Electric guitar", "Synthesizer", "Soft", "Energetic", "Electronic", "Self-discovery",
    "Sad", "Ballad", "Longing", "Meditation", "Faith", "Acoustic", "Peaceful", "Wedding",
    "Piano", "Strings", "Acoustic guitar", "Romantic", "Drums", "Emotional", "Walking",
    "Hope", "Hopeful", "Powerful", "Epic", "Driving", "Rock"
]

class LLMProvider(ABC):
    @abstractmethod
    def generate_text(self, prompt: str, model: str, **kwargs) -> str:
        pass

    @abstractmethod
    def generate_json(self, prompt: str, model: str, **kwargs) -> Dict:
        pass
    
    @abstractmethod
    def get_models(self) -> List[str]:
        pass

class OllamaProvider(LLMProvider):
    def __init__(self, base_url: str):
        self.base_url = base_url

    def get_models(self) -> List[str]:
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=2)
            if resp.status_code == 200:
                data = resp.json()
                return [model["name"] for model in data.get("models", [])]
        except Exception as e:
            logger.warning(f"Failed to fetch Ollama models: {e}")
        return []

    def generate_text(self, prompt: str, model: str, **kwargs) -> str:
        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": kwargs.get("options", {})
                },
                timeout=kwargs.get("timeout", 60)
            )
            if resp.status_code == 200:
                return resp.json().get("response", "")
            else:
                raise Exception(f"Ollama Error: {resp.text}")
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            raise e

    def generate_json(self, prompt: str, model: str, **kwargs) -> Dict:
        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                    "options": kwargs.get("options", {})
                },
                timeout=kwargs.get("timeout", 60)
            )
            if resp.status_code == 200:
                raw_response = resp.json().get("response", "")
                raw_response = self._clean_json(raw_response)
                return json.loads(raw_response)
            else:
                raise Exception(f"Ollama Error: {resp.text}")
        except Exception as e:
            logger.error(f"Ollama JSON generation failed: {e}")
            raise e

    def _clean_json(self, raw_response: str) -> str:
        raw_response = raw_response.strip()
        if raw_response.startswith("```json"):
            raw_response = raw_response.replace("```json", "").replace("```", "")
        elif raw_response.startswith("```"):
             raw_response = raw_response.replace("```", "")
        return raw_response

class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str, base_url: Optional[str] = None):
        if OpenAI is None:
            raise ImportError("OpenAI library is not installed. Please run `pip install openai`.")
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def get_models(self) -> List[str]:
        try:
            # Iterate directly to handle pagination automatically
            return [model.id for model in self.client.models.list()]
        except Exception as e:
            logger.warning(f"Failed to fetch OpenAI models: {e}")
            return []

    def generate_text(self, prompt: str, model: str, **kwargs) -> str:
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get("options", {}).get("temperature", 0.7),
            )
            return response.choices[0].message.content
        except Exception as e:
            # Check for OpenRouter invalid model error (400)
            is_openrouter = self.client.base_url.host == "openrouter.ai" or "openrouter.ai" in str(self.client.base_url)
            if is_openrouter and "400" in str(e):
                logger.warning(f"OpenRouter model {model} failed (likely invalid/deprecated). Attempting fallback to free model.")
                try:
                    fallback_model = "google/gemini-2.0-flash-exp:free"
                    response = self.client.chat.completions.create(
                        model=fallback_model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=kwargs.get("options", {}).get("temperature", 0.7),
                    )
                    return response.choices[0].message.content
                except Exception as fallback_e:
                    logger.error(f"OpenRouter fallback failed: {fallback_e}")
                    raise e # Raise original error if fallback fails
            
            logger.error(f"OpenAI generation failed: {e}")
            raise e

    def generate_json(self, prompt: str, model: str, **kwargs) -> Dict:
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=kwargs.get("options", {}).get("temperature", 0.7),
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            # Check for OpenRouter invalid model error (400)
            is_openrouter = self.client.base_url.host == "openrouter.ai" or "openrouter.ai" in str(self.client.base_url)
            if is_openrouter and "400" in str(e):
                logger.warning(f"OpenRouter model {model} failed (likely invalid/deprecated). Attempting fallback to free model.")
                try:
                    fallback_model = "google/gemini-2.0-flash-exp:free"
                    response = self.client.chat.completions.create(
                        model=fallback_model,
                        messages=[{"role": "user", "content": prompt}],
                        response_format={"type": "json_object"},
                        temperature=kwargs.get("options", {}).get("temperature", 0.7),
                    )
                    return json.loads(response.choices[0].message.content)
                except Exception as fallback_e:
                    logger.error(f"OpenRouter fallback failed: {fallback_e}")
                    raise e
            
            logger.error(f"OpenAI JSON generation failed: {e}")
            raise e

class GeminiProvider(LLMProvider):
    def __init__(self, api_key: str):
        if genai is None:
             raise ImportError("Google GenAI library is not installed. Please run `pip install google-genai`.")
        self.client = genai.Client(api_key=api_key)

    def get_models(self) -> List[str]:
        try:
            # New SDK model listing
            # Models are yielded. We extract the 'name' or use 'display_name'
            # Usually names are like 'models/gemini-1.5-flash'
            models = []
            for m in self.client.models.list():
                 # Filter somewhat if possible, but SDK might not expose 'supported_generation_methods' directly on the iterator object easily without inspection
                 # For now, just list them. The name usually comes with 'models/' prefix, user might want short name?
                 # Let's keep full name 'models/...' or strip it. Old logic kept name.
                 models.append(m.name.replace('models/', '') if m.name.startswith('models/') else m.name)
            return models
        except Exception as e:
             logger.warning(f"Failed to fetch Gemini models: {e}")
             return ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-1.0-pro"]

    def generate_text(self, prompt: str, model: str, **kwargs) -> str:
        try:
            # New SDK generation
            response = self.client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=kwargs.get("options", {}).get("temperature", 0.7)
                )
            )
            return response.text
        except Exception as e:
            logger.error(f"Gemini generation failed: {e}")
            raise e

    def generate_json(self, prompt: str, model: str, **kwargs) -> Dict:
        try:
            # New SDK JSON enforcement
            response = self.client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type='application/json',
                    temperature=kwargs.get("options", {}).get("temperature", 0.7)
                )
            )
            return json.loads(response.text)
        except Exception as e:
             # Fallback manual clean if SDK fails to enforce pure JSON for some reason (unlikely with this config)
            logger.error(f"Gemini JSON generation failed: {e}")
            raise e



class LLMService:
    @staticmethod
    def _get_provider(override_config: Optional[Dict] = None) -> LLMProvider:
        """
        Get provider instance. 
        If override_config is provided (for testing credentials), use that. 
        Otherwise use ConfigManager.
        """
        if override_config:
            config = override_config
            provider_name = config.get("provider", "ollama")
            # For override, we expect structure like { "provider": "openai", "openai": { ... } }
            # Or just flat if simpler, but let's stick to config structure
        else:
            config = ConfigManager().get_config()
            provider_name = config.get("provider", "ollama")
        
        if provider_name == "ollama":
            base_url = config.get("ollama", {}).get("base_url", "http://localhost:11434")
            return OllamaProvider(base_url=base_url)
        elif provider_name == "openai":
            api_key = config.get("openai", {}).get("api_key", "")
            return OpenAIProvider(api_key=api_key)
        elif provider_name == "deepseek":
            api_key = config.get("deepseek", {}).get("api_key", "")
            return OpenAIProvider(
                api_key=api_key, 
                base_url="https://api.deepseek.com"
            )
        elif provider_name == "openrouter":
            api_key = config.get("openrouter", {}).get("api_key", "")
            return OpenAIProvider(
                api_key=api_key, 
                base_url="https://openrouter.ai/api/v1"
            )
        elif provider_name == "lmstudio":
            base_url = config.get("lmstudio", {}).get("base_url", "http://localhost:1234/v1")
            return OpenAIProvider(
                api_key="lm-studio", 
                base_url=base_url
            )
        elif provider_name == "gemini":
            api_key = config.get("gemini", {}).get("api_key", "")
            return GeminiProvider(api_key=api_key)
        else:
            return OllamaProvider(base_url="http://localhost:11434")

    @staticmethod
    def fetch_available_models(provider_name: str, api_key: Optional[str] = None, base_url: Optional[str] = None) -> List[str]:
        """
        Fetch available models for a given provider using provided credentials.
        This is used for the settings dropdown to test connection and list models.
        """
        try:
            # Construct a temporary config to instantiate the provider
            temp_config = {"provider": provider_name}
            
            # Map specific args to the config structure expected by _get_provider
            if provider_name == "ollama":
                temp_config["ollama"] = {"base_url": base_url or "http://localhost:11434"}
            elif provider_name == "openai":
                temp_config["openai"] = {"api_key": api_key}
            elif provider_name == "deepseek":
                temp_config["deepseek"] = {"api_key": api_key}
            elif provider_name == "openrouter":
                temp_config["openrouter"] = {"api_key": api_key} 
            elif provider_name == "lmstudio":
                temp_config["lmstudio"] = {"base_url": base_url or "http://localhost:1234/v1"}
            elif provider_name == "gemini":
                temp_config["gemini"] = {"api_key": api_key}
                
            provider = LLMService._get_provider(override_config=temp_config)
            return provider.get_models()
        except Exception as e:
            logger.error(f"Failed to fetch models for {provider_name}: {e}")
            raise e

    @staticmethod
    def get_models() -> List[str]:
        return LLMService._get_provider().get_models()

    @staticmethod
    def _get_active_model() -> str:
        config = ConfigManager().get_config()
        provider = config.get("provider", "ollama")
        return config.get(provider, {}).get("model", "llama3")

    @staticmethod
    def generate_lyrics(topic: str, model: Optional[str] = None, seed_lyrics: Optional[str] = None) -> str:
        provider = LLMService._get_provider()
        model = model or LLMService._get_active_model()

        if seed_lyrics and seed_lyrics.strip():
            prompt = (
                f"Continue and complete these song lyrics. Topic/Context: {topic}.\n"
                f"EXISTING LYRICS (Keep these exactly as is, and append the rest):\n"
                f"'''{seed_lyrics}'''\n\n"
                "INSTRUCTIONS:\n"
                "1. Keep the existing lyrics at the start.\n"
                "2. Generate the missing parts to complete a full song structure (Intro, Verse, Chorus, Bridge, Outro).\n"
                "3. Ensure strictly formatted with tags [Verse], [Chorus] etc.\n"
                "4. Do NOT output any conversational text, ONLY the lyrics.\n"
            )
        else:
            prompt = (
                f"Write song lyrics about: {topic}. "
                "IMPORTANT: Use the following format strictly:\n"
                "[Intro]\n\n"
                "[Verse]\n"
                "(lyrics here)\n\n"
                "[Chorus]\n"
                "(lyrics here)\n\n"
                "[Bridge]\n"
                "(lyrics here)\n\n"
                "[Outro]\n\n"
                "Do not include any conversational filler. Just the formatted lyrics."
            )
        
        return provider.generate_text(prompt, model)

    @staticmethod
    def generate_title(context: str, model: Optional[str] = None) -> str:
        provider = LLMService._get_provider()
        model = model or LLMService._get_active_model()
        
        prompt = f"Generate a short, creative, 2-5 word song title based on this concept/lyrics: '{context}'. Return ONLY the title, no quotes or prefix."
        
        try:
            return provider.generate_text(prompt, model).strip().replace('"', '')
        except Exception:
            return "Untitled Track"
            
    @staticmethod
    def enhance_prompt(concept: str, model: Optional[str] = None) -> dict:
        provider = LLMService._get_provider()
        model = model or LLMService._get_active_model()

        valid_tags_str = ", ".join(VALID_HEARTMULA_TAGS)
        prompt = (
            f"Act as a professional music producer. Transform this simple user concept into a detailed musical direction.\n"
            f"USER CONCEPT: '{concept}'\n\n"
            "INSTRUCTIONS:\n"
            "1. Create a 'topic' description that is evocative and detailed (1 sentence).\n"
            f"2. Select 3-5 'tags' ONLY from this list: [{valid_tags_str}]. Do NOT use any other tags.\n"
            "3. Return ONLY a raw JSON object with keys 'topic' and 'tags'. Do NOT wrap in markdown code blocks.\n\n"
            "Example Output:\n"
            '{"topic": "A melancholic acoustic ballad about lost love in autumn.", "tags": "Acoustic, Sad, Soft"}'
        )

        try:
            return provider.generate_json(prompt, model)
        except Exception as e:
            logger.warning(f"Enhance prompt failed: {e}")
            return {"topic": concept, "tags": "Pop, Soft"}

    @staticmethod
    def generate_inspiration(model: Optional[str] = None) -> dict:
        provider = LLMService._get_provider()
        model = model or LLMService._get_active_model()

        valid_tags_str = ", ".join(VALID_HEARTMULA_TAGS)
        prompt = (
            "Act as a professional music producer brainstorming new hit songs.\n"
            "INSTRUCTIONS:\n"
            "1. Invent a UNIQUE, creative song concept/topic (1 vivid sentence).\n"
            f"2. Select a matching musical style using 3-5 tags ONLY from this list: [{valid_tags_str}].\n"
            "3. Return ONLY a raw JSON object with keys 'topic' and 'tags'.\n\n"
            "Examples:\n"
            '{"topic": "A lonely astronaut drifting through the cosmos.", "tags": "Reflection, Space, Soft"}\n'
            '{"topic": "A cyberpunk detective chasing a suspect in rain.", "tags": "Electronic, Dark, Driving"}'
        )

        try:
             # High creativity handled by providers implicitly or via temperature in kwargs if exposed
            return provider.generate_json(prompt, model, options={"temperature": 0.9})
        except Exception as e:
            logger.warning(f"Inspiration generation failed: {e}")
            return {"topic": "A mysterious journey through time", "tags": "Strings, Epic, Cinematic"}

    @staticmethod
    def generate_styles_list(model: Optional[str] = None) -> List[str]:
        try:
            return random.sample(VALID_HEARTMULA_TAGS, 12)
        except Exception:
            return VALID_HEARTMULA_TAGS[:12]

    @staticmethod
    def update_config(provider_name: str, config_data: Dict[str, Any]):
        ConfigManager().update_config({provider_name: config_data})

    @staticmethod
    def set_active_provider(provider_name: str):
        ConfigManager().set_provider(provider_name)
    
    @staticmethod
    def get_config() -> Dict[str, Any]:
        return ConfigManager().get_config()
