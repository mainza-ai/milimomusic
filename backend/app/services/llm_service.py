import requests
import logging
from typing import List, Optional, Dict, Any, Type
import json
import random
from abc import ABC, abstractmethod
import os
from datetime import datetime
from pydantic import BaseModel

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
from .lyrics_schemas import LyricsResponse
from .lyrics_engine import StructuredLyricsEngine
from .lyrics_utils import LyricsDOM
from .style_registry import StyleRegistry, OFFICIAL_STYLES

logger = logging.getLogger(__name__)

# Legacy alias for backward compatibility
VALID_HEARTMULA_TAGS = OFFICIAL_STYLES

class LLMProvider(ABC):
    @abstractmethod
    def generate_text(self, prompt: str, model: str, **kwargs) -> str:
        pass

    @abstractmethod
    def generate_json(self, prompt: str, model: str, **kwargs) -> Dict:
        pass
    
    @abstractmethod
    def generate_structured(self, prompt: str, model: str, response_format: Type[BaseModel], **kwargs) -> BaseModel:
        """Generates a structured Pydantic object."""
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
                timeout=kwargs.get("timeout", 300)
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
                timeout=kwargs.get("timeout", 300)
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
            
    def generate_structured(self, prompt: str, model: str, response_format: Type[BaseModel], **kwargs) -> BaseModel:
        # Ollama doesn't natively support client.parse-like schema enforcement yet (except via generic JSON mode).
        # We generate JSON and validate with Pydantic.
        json_data = self.generate_json(prompt, model, **kwargs)
        return response_format.model_validate(json_data)

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
            self._handle_error(e, model)
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
            self._handle_error(e, model)
            raise e

    def generate_structured(self, prompt: str, model: str, response_format: Type[BaseModel], **kwargs) -> BaseModel:
        try:
            # Use beta parse if available and robust
            completion = self.client.beta.chat.completions.parse(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that outputs strict structured JSON."},
                    {"role": "user", "content": prompt}
                ],
                response_format=response_format,
                temperature=kwargs.get("options", {}).get("temperature", 0.7),
            )
            parsed = completion.choices[0].message.parsed
            if not parsed:
                 raise ValueError("Failed to parse structured output from OpenAI response.")
            return parsed
        except Exception as e:
            self._handle_error(e, model)
            # If parse fails (e.g. model doesn't support generic parse), fallback to JSON
            logger.warning(f"Generate structured failed, falling back to JSON mode: {e}")
            json_data = self.generate_json(prompt, model, **kwargs)
            return response_format.model_validate(json_data)

    def _handle_error(self, e, model):
         # Check for OpenRouter invalid model error (400)
        is_openrouter = self.client.base_url.host == "openrouter.ai" or "openrouter.ai" in str(self.client.base_url)
        if is_openrouter and "400" in str(e):
            logger.warning(f"OpenRouter model {model} failed. This might be due to model deprecation.")

class GeminiProvider(LLMProvider):
    def __init__(self, api_key: str):
        if genai is None:
             raise ImportError("Google GenAI library is not installed. Please run `pip install google-genai`.")
        self.client = genai.Client(api_key=api_key)

    def get_models(self) -> List[str]:
        try:
            models = []
            for m in self.client.models.list():
                 models.append(m.name.replace('models/', '') if m.name.startswith('models/') else m.name)
            return models
        except Exception as e:
             logger.warning(f"Failed to fetch Gemini models: {e}")
             return ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-1.0-pro"]

    def generate_text(self, prompt: str, model: str, **kwargs) -> str:
        try:
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
            logger.error(f"Gemini JSON generation failed: {e}")
            raise e
            
    def generate_structured(self, prompt: str, model: str, response_format: Type[BaseModel], **kwargs) -> BaseModel:
         # Gemini SDK supports specific schema or just JSON. Fallback to JSON + Pydantic.
         # Future: Use `response_schema` in config if supported by pydantic mapping.
         json_data = self.generate_json(prompt, model, **kwargs)
         return response_format.model_validate(json_data)



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
        try:
            temp_config = {"provider": provider_name}
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
        provider_name = config.get("provider", "ollama")
        model = config.get(provider_name, {}).get("model")

        # Smart fallback for Ollama: If configured model is missing/default, pick distinct available one
        if provider_name == "ollama":
            try:
                base_url = config.get("ollama", {}).get("base_url", "http://localhost:11434")
                # Quick check without massive overhead (timeout is 2s in get_models)
                provider = OllamaProvider(base_url=base_url)
                available = provider.get_models()
                
                if available:
                    # If no model configured, or configured default is NOT in available, pick first available
                    if not model or model not in available:
                        logger.info(f"Auto-switching Ollama model from '{model}' to '{available[0]}'")
                        return available[0]
            except Exception as e:
                logger.warning(f"Failed to auto-detect Ollama model: {e}")

        return model or "llama3.2:3b-instruct-fp16"

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
                "1. START with the Existing Lyrics. You must incorporate them into the first section (e.g. [Intro] or [Verse 1]).\n"
                "   - WRONG: 'I saw a UFO\\n\\n[Verse 1]...'\n"
                "   - CORRECT: '[Verse 1]\\nI saw a UFO\\n...'\n"
                "2. Generate the missing parts to complete a full song structure.\n"
                "3. Ensure strictly formatted with tags [Verse], [Chorus] etc.\n"
                "5. FORMATTING: Output ONLY lyrics. NO stage directions like '(guitar solo)', '(instrumental)', or '(repeat chorus)'.\n"
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
                "RULES:\n"
                "- Do not include any conversational filler. Just the formatted lyrics.\n"
                "- FORMATTING: Output ONLY lyrics. NO stage directions like '(guitar solo)', '(instrumental)', or '(repeat chorus)'."
            )
        
        response = provider.generate_text(prompt, model)
        
        try:
            with open("ai_debug.log", "a") as f:
                f.write(f"\n\n--- INITIAL GENERATION ({datetime.now().isoformat()}) ---\n")
                f.write(f"PROMPT:\n{prompt}\n")
                f.write(f"RESPONSE:\n{response}\n")
        except Exception as e:
            print(f"Failed to write to debug log: {e}")
            
        return response

    @staticmethod
    async def generate_lyrics_async(topic: str, model: Optional[str] = None, seed_lyrics: Optional[str] = None, tags: Optional[str] = None) -> str:
        """
        Async version of generate_lyrics that uses the pydantic-graph.
        Mode = CREATION
        """
        from .lyrics_graph import run_lyrics_graph
        
        provider = LLMService._get_provider()
        model = model or LLMService._get_active_model()
        
        try:
            # Run Graph with correct signature
            result = await run_lyrics_graph(
                current_lyrics=seed_lyrics or "",
                user_message="Write a full song based on the topic and style.",  # Implicit request
                topic=topic,
                tags=tags or "Any",
                provider=provider,
                model_name=model
            )
            
            if result and result.get("lyrics"):
                return result["lyrics"]
            else:
                return seed_lyrics or "Generation failed."
                
        except Exception as e:
            logger.error(f"Generate lyrics async failed: {e}")
            raise e

    @staticmethod
    def chat_with_lyrics(current_lyrics: str, user_message: str, model: Optional[str] = None, chat_history: Optional[List[Dict[str, Any]]] = None, topic: Optional[str] = None, tags: Optional[str] = None) -> Dict[str, str]:
        provider = LLMService._get_provider()
        model = model or LLMService._get_active_model()
        
        # Analyze Structure
        dom = LyricsDOM(current_lyrics)
        structure_map = dom.get_structure_map()
        
        # SHORT LYRICS BYPASS
        if len(current_lyrics) < 150 or current_lyrics.count('\n') < 3:
            logger.info("Short lyrics detected. Bypassing Structured Engine for full generation.")
            context_header = ""
            if topic: context_header += f"Overall Topic: {topic}. "
            if tags: context_header += f"Style: {tags}. "
            
            combined_prompt = f"{context_header}\nOriginal idea: {current_lyrics}\nUser feedback: {user_message}"
            full_song = LLMService.generate_lyrics(topic=combined_prompt, model=model, seed_lyrics=current_lyrics)
            
            return {
                "message": "I've fleshed out your idea into a full song.",
                "lyrics": full_song
            }

        context_str = ""
        if topic: context_str += f"SONG CONCEPT: {topic}\n"
        if tags: context_str += f"STYLE/GENRE: {tags}\n"

        # STRUCTURED PROMPT
        prompt = (
            "ROLE: You are an award-winning professional songwriter and lyricist.\n"
            "GOAL: Update the lyrics based on the user's request.\n"
            f"{context_str}"
            "MECHANISM: You do not output raw text. You output a JSON object with a LIST OF OPERATIONS.\n\n"
            f"CURRENT STRUCTURE MAP: {structure_map}\n"
            f"CURRENT LYRICS CONTENT:\n'''{current_lyrics}'''\n\n"
            f"USER REQUEST: \"{user_message}\"\n\n"
            "INSTRUCTIONS for Operations:\n"
            "1. UPDATE_SECTION: Re-write an existing section. NOTE: This REPLACES the entire section content.\n"
            "2. APPEND_CONTENT: Add lines to the END of an existing section. Safer for 'adding a line'.\n"
            "3. INSERT_SECTION: Add a NEW section. specify 'insert_position' (BEFORE/AFTER) relative to the target.\n"
            "4. DELETE_SECTION: Remove a section.\n\n"
            "REQUIRED JSON OUTPUT FORMAT:\n"
            "{\n"
            "  \"thought_process\": \"Brief explanation of your plan...\",\n"
            "  \"operations\": [\n"
            "    {\n"
            "      \"op_type\": \"UPDATE_SECTION\",\n"
            "      \"target_section_type\": \"Verse\",\n"
            "      \"target_section_index\": 1,\n"
            "      \"new_content\": \"updated lines...\"\n"
            "    },\n"
            "    {\n"
            "      \"op_type\": \"INSERT_SECTION\",\n"
            "      \"target_section_type\": \"Chorus\",\n"
            "      \"insert_position\": \"AFTER\",\n"
            "      \"new_section_type\": \"Bridge\",\n"
            "      \"new_content\": \"lines...\"\n"
            "    }\n"
            "  ]\n"
            "}\n\n"
            "RULES:\n"
            "- MINIMAL CHANGES: Touch ONLY the sections the user explicitly asked to change. Leave ALL other sections exactly as they are.\n"
            "- NO HALLUCINATED UPDATES: Do NOT 'improve' or 'rewrite' sections unless asked.\n"
            "- ADDING SECTIONS:\n"
            "  - If user says 'Add an Intro', use `INSERT_SECTION` with `target_section_type='Verse'`, `target_section_index=1` and `insert_position='BEFORE'`.\n"
            "  - NEVER use `UPDATE_SECTION` to add a new section (this overwrites existing content).\n"
            "- ALWAYS provide full new content for updates.\n"
            "- Do NOT hallucinate section indices. Use the Structure Map provided.\n"
            "- CONTEXT: New lines MUST match the rhyme scheme, meter, and theme of the surrounding lines.\n"
            "- FORMATTING: Output ONLY lyrics. NO stage directions like '(guitar solo)', '(instrumental)', or '(repeat chorus)'.\n"
            "- CONTENT CLEANLINESS: The `new_content` field must contain lyrics ONLY. Do NOT include the section header (e.g. \"[Verse 1]\") inside `new_content`. The system adds this automatically.\n"
            "- DELETING LINES: To remove a line from a section, use UPDATE_SECTION with the lines you want to KEEP. Do NOT use DELETE_SECTION unless removing the ENTIRE section.\n"
        )

        try:
            # DEBUG: Log INPUT
            try:
                with open("ai_debug.log", "a") as f:
                    timestamp = datetime.now().isoformat()
                    f.write(f"\n\n=== NEW REQUEST ({timestamp}) ===\n")
                    f.write(f"USER MESSAGE: {user_message}\n")
                    f.write(f"CONTEXT TOPIC: {topic} | TAGS: {tags}\n")
                    f.write(f"STRUCTURE MAP: {structure_map}\n")
                    f.write(f"CURRENT LYRICS ({len(current_lyrics)} chars):\n{current_lyrics[:200]}...\n")
                    f.write("--------------------------------\n")
            except Exception as log_e:
                print(f"Logging failed: {log_e}")

            # Generate Structured Plan
            result: LyricsResponse = provider.generate_structured(prompt, model, LyricsResponse, options={"temperature": 0.4})
            
            # Log
            debug_msg = f"--- STRUCTURED ENGINE RESPONSE ---\nThought: {result.thought_process}\nOps: {len(result.operations)}\n"
            print(debug_msg)
            try:
                with open("ai_debug.log", "a") as f:
                    f.write(debug_msg)
                    f.write(f"{result.model_dump_json(indent=2)}\n")
            except: pass
            
            engine = StructuredLyricsEngine()
            new_lyrics = engine.apply_edits(current_lyrics, result.operations)
            
            return {
                "message": result.thought_process,
                "lyrics": new_lyrics
            }
            
        except Exception as e:
            logger.error(f"Lyrics chat failed: {e}")
            return {
                "message": "I encountered an error processing your request. Please try again.",
                "lyrics": current_lyrics
            }

    @staticmethod
    async def chat_with_lyrics_async(
        current_lyrics: str, 
        user_message: str, 
        model: Optional[str] = None, 
        chat_history: Optional[List[Dict[str, Any]]] = None, 
        topic: Optional[str] = None, 
        tags: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Async version using pydantic-graph multi-agent architecture.
        
        Features:
        - Automatic retry on LLM failures (up to 3 attempts)
        - Separate Lyricist and StructureGuard agents
        - Persistent SongState through the graph
        """
        from .lyrics_graph import run_lyrics_graph, MaxRetriesExceededError
        
        provider = LLMService._get_provider()
        model = model or LLMService._get_active_model()
        
        # Debug logging - INITIAL STATE
        try:
            with open("ai_debug.log", "a") as f:
                timestamp = datetime.now().isoformat()
                f.write(f"\n\n{'='*60}\n")
                f.write(f"=== LYRICS CHAT REQUEST ({timestamp}) ===\n")
                f.write(f"{'='*60}\n\n")
                f.write(f"USER MESSAGE: {user_message}\n\n")
                f.write(f"CONTEXT: Topic='{topic}' | Tags='{tags}'\n\n")
                f.write(f"--- INITIAL LYRICS ({len(current_lyrics)} chars) ---\n")
                f.write(f"{current_lyrics}\n")
                f.write(f"--- END INITIAL LYRICS ---\n\n")
        except Exception as log_e:
            logger.warning(f"Logging failed: {log_e}")
        
        try:
            result = await run_lyrics_graph(
                current_lyrics=current_lyrics,
                user_message=user_message,
                topic=topic,
                tags=tags,
                provider=provider,
                model_name=model,
            )
            
            # Debug log success with FINAL LYRICS
            try:
                with open("ai_debug.log", "a") as f:
                    f.write(f"--- GRAPH SUCCESS ---\n")
                    f.write(f"AI Message: {result.get('message', 'N/A')}\n\n")
                    new_lyrics = result.get('lyrics', '')
                    f.write(f"--- NEW LYRICS ({len(new_lyrics)} chars) ---\n")
                    f.write(f"{new_lyrics}\n")
                    f.write(f"--- END NEW LYRICS ---\n")
                    f.write(f"{'='*60}\n\n")
            except:
                pass
            
            return result
            
        except MaxRetriesExceededError as e:
            logger.error(f"Graph max retries exceeded: {e}")
            return {
                "message": str(e),
                "lyrics": current_lyrics,
                "error": True
            }
        except Exception as e:
            logger.error(f"Lyrics graph failed: {e}")
            return {
                "message": f"An error occurred: {str(e)}",
                "lyrics": current_lyrics,
                "error": True
            }

    @staticmethod
    def generate_title(context: str, model: Optional[str] = None) -> str:
        provider = LLMService._get_provider()
        model = model or LLMService._get_active_model()
        
        prompt = f"Generate a short, creative, 2-5 word song title based on this concept/lyrics: '{context}'. Return ONLY the title, no quotes or prefix."
        
        try:
            response = provider.generate_text(prompt, model).strip().replace('"', '')
            return response
        except Exception:
            return "Untitled Track"
            
    @staticmethod
    def enhance_prompt(concept: str, model: Optional[str] = None) -> dict:
        provider = LLMService._get_provider()
        model = model or LLMService._get_active_model()

        # Dynamic style fetching
        valid_tags = StyleRegistry().get_styles_for_prompt()
        valid_tags_str = ", ".join(valid_tags)
        
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
            result = provider.generate_json(prompt, model)
            return result
        except Exception as e:
            logger.warning(f"Enhance prompt failed: {e}")
            return {"topic": concept, "tags": "Pop, Soft"}

    @staticmethod
    def generate_inspiration(model: Optional[str] = None) -> dict:
        provider = LLMService._get_provider()
        model = model or LLMService._get_active_model()

        # Dynamic style fetching
        valid_tags = StyleRegistry().get_styles_for_prompt()
        valid_tags_str = ", ".join(valid_tags)

        prompt = (
            "Act as a professional music producer brainstorming new hit songs.\n"
            "INSTRUCTIONS:\n"
            "1. Invent a UNIQUE, creative song concept/topic (1 vivid sentence).\n"
            f"2. Select a matching musical style using 3-5 tags ONLY from this list: [{valid_tags_str}].\n"
            "3. Return ONLY a raw JSON object with keys 'topic' and 'tags'.\n"
            "4. IMPORTANT: Do NOT use any tags not in the list above!\n\n"
            "Examples:\n"
            '{"topic": "A lonely astronaut drifting through the cosmos.", "tags": "Reflection, Soft, Emotional"}\n'
            '{"topic": "A cyberpunk detective chasing a suspect in rain.", "tags": "Electronic, Driving, Synthesizer"}'
        )

        try:
            result = provider.generate_json(prompt, model, options={"temperature": 0.9})
            
            # Post-validation: filter out any invalid tags the AI might have hallucinated
            if "tags" in result:
                tags_str = result["tags"]
                if isinstance(tags_str, str):
                    # Split tags by comma or comma-space
                    raw_tags = [t.strip() for t in tags_str.replace(", ", ",").split(",")]
                    # Filter to only valid tags (case-insensitive matching)
                    # Use dynamic list for validation
                    all_styles = StyleRegistry().get_styles_for_prompt()
                    valid_lower = {t.lower(): t for t in all_styles}
                    
                    valid_tags = [valid_lower.get(t.lower(), None) for t in raw_tags]
                    valid_tags = [t for t in valid_tags if t is not None]
                    
                    if not valid_tags:
                        valid_tags = ["Pop", "Soft", "Emotional"]  # Fallback
                    
                    result["tags"] = ", ".join(valid_tags)
                    logger.info(f"Inspiration tags filtered: {raw_tags} -> {valid_tags}")
            
            return result
        except Exception as e:
            logger.warning(f"Inspiration generation failed: {e}")
            return {"topic": "A mysterious journey through time", "tags": "Strings, Epic, Emotional"}

    @staticmethod
    def generate_styles_list(model: Optional[str] = None) -> List[str]:
        """Get a random sample of available styles from the registry."""
        try:
            registry = StyleRegistry()
            all_styles = registry.get_styles_for_prompt()
            return random.sample(all_styles, min(12, len(all_styles)))
        except Exception:
            return OFFICIAL_STYLES[:12]

    @staticmethod
    def update_config(provider_name: str, config_data: Dict[str, Any]):
        ConfigManager().update_config({provider_name: config_data})

    @staticmethod
    def set_active_provider(provider_name: str):
        ConfigManager().set_provider(provider_name)
    
    @staticmethod
    def get_config() -> Dict[str, Any]:
        return ConfigManager().get_config()
