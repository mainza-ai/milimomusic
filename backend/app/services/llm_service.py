import asyncio
import requests
import logging
from typing import List, Optional, Dict, Any, Type
import json
import re
import random
import time
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
from app.core.llm_contracts import LLMResult, classify_llm_error

logger = logging.getLogger(__name__)

# Legacy alias for backward compatibility
VALID_HEARTMULA_TAGS = OFFICIAL_STYLES


def _strip_thinking(text: str) -> str:
    """Remove model reasoning/thinking tokens so only the final answer remains.

    Handles all common CoT / reasoning envelopes used by DeepSeek R1, Qwen,
    Ollama, Anthropic, Gemini, OpenAI, and local LLMs:
    - <think>...</think>, <thinking>...</thinking>, <reasoning>...</reasoning>
    - Orphaned closing tags (e.g. `...notes...</think>`)
    - Orphaned opening tags (e.g. `<think>...unclosed`)
    - Markdown scratchpad headers: `### Thinking Process`, `**Analysis:**`, `##### reasoning #####`
    - `::` delimited thinking blocks
    """
    if not text:
        return ""
    if not isinstance(text, str):
        return text
    out = text

    # 1. Standard matched pairs of thinking/reasoning tags
    tag_names = r"think|thinking|reasoning|analysis|thought|reflection|deliberation|scratchpad|internal_thought"
    out = re.sub(
        rf"<(?:\s*{tag_names})[^>]*>.*?</(?:\s*{tag_names})>",
        "",
        out,
        flags=re.DOTALL | re.IGNORECASE
    )

    # 2. Orphaned closing tags: if a closing </think> appears with content before it,
    # drop everything up to and including the closing tag
    out = re.sub(
        rf"^.*?</(?:\s*{tag_names})>\s*",
        "",
        out,
        flags=re.DOTALL | re.IGNORECASE
    )

    # 3. Unmatched/unclosed opening tags: e.g. <think> at the start before a section
    out = re.sub(
        rf"<\s*(?:{tag_names})[^>]*>.*?(?=\n\s*\[|\Z)",
        "",
        out,
        flags=re.DOTALL | re.IGNORECASE
    )

    # 4. OpenAI/Anthropic/Custom delimiters
    out = re.sub(r"<[^>]*response[^>]*>", "\n", out, flags=re.IGNORECASE)
    out = re.sub(r"<[^>]*deliberation[^>]*>", "\n", out, flags=re.IGNORECASE)

    # 5. Markdown reasoning headers & blocks
    out = re.sub(r"#####\s*reasoning\s*#####.*?(?=\n|$)", "", out, flags=re.DOTALL | re.IGNORECASE)
    out = re.sub(r"(?i)###\s*(?:thinking|thought|scratchpad|reasoning|analysis).*?(?=\n\s*\[|\Z)", "", out, flags=re.DOTALL)
    out = re.sub(r"(?i)\*\*(?:thinking|thought process|analysis|reasoning):\*\*.*?(?=\n\s*\[|\Z)", "", out, flags=re.DOTALL)

    # 6. :: delimited thinking blocks (":: emotion ...\n\n answer")
    out = re.sub(r"^\s*::.*?(\n|$)", "", out, flags=re.MULTILINE)

    # 7. Stray closing tags remaining anywhere
    out = re.sub(rf"</(?:\s*{tag_names})>", "", out, flags=re.IGNORECASE)

    return out.strip()

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
                timeout=kwargs.get("timeout", (3.0, 30.0))
            )
            if resp.status_code == 200:
                return _strip_thinking(resp.json().get("response", ""))
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
                timeout=kwargs.get("timeout", (3.0, 30.0))
            )
            if resp.status_code == 200:
                raw_response = _strip_thinking(resp.json().get("response", ""))
                # Extract the outermost JSON object if prose/thinking preceded it
                start = raw_response.find("{")
                end = raw_response.rfind("}")
                if start != -1 and end != -1 and end > start:
                    raw_response = raw_response[start:end + 1]
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

    def generate_chat(self, messages: List[Dict[str, str]], model: str, **kwargs) -> LLMResult:
        """Native Ollama /api/chat: real multi-turn roles + system prompt.

        Returns an LLMResult envelope carrying token usage (prompt_eval_count /
        eval_count) instead of dropping it.
        """
        started = time.monotonic()
        temperature = kwargs.get("options", {}).get("temperature", 0.7)
        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": model,
                    "messages": messages,
                    "stream": False,
                    "options": kwargs.get("options", {"temperature": temperature}),
                },
                timeout=kwargs.get("timeout", (3.0, 60.0)),
            )
            if resp.status_code != 200:
                raise Exception(f"Ollama Error ({resp.status_code}): {resp.text}")
            data = resp.json()
            content = _strip_thinking(data.get("message", {}).get("content", ""))
            usage = {}
            if data.get("prompt_eval_count"):
                usage["prompt_tokens"] = int(data["prompt_eval_count"])
            if data.get("eval_count"):
                usage["completion_tokens"] = int(data["eval_count"])
            return LLMResult(
                content=content,
                provider="ollama",
                model=model,
                latency_ms=int((time.monotonic() - started) * 1000),
                usage=usage,
            )
        except Exception as e:
            raise classify_llm_error("ollama", e) from e

class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str, base_url: Optional[str] = None, timeout: float = 30.0):
        if OpenAI is None:
            raise ImportError("OpenAI library is not installed. Please run `pip install openai`.")
        self.client = OpenAI(api_key=api_key or "no-key", base_url=base_url, timeout=timeout, max_retries=2)

    def get_models(self) -> List[str]:
        try:
            # Iterate directly to handle pagination automatically
            models = [model.id for model in self.client.models.list()]
            return models
        except Exception as e:
            logger.warning(f"Failed to fetch OpenAI models: {e}")
            if self.client.base_url and "nvidia.com" in str(self.client.base_url):
                return [
                    "deepseek-ai/deepseek-v4-flash-0731",
                    "deepseek-ai/deepseek-r1",
                    "qwen/qwen2.5-72b-instruct",
                    "mistralai/mistral-large-2-instruct",
                ]
            elif self.client.base_url and "deepseek.com" in str(self.client.base_url):
                return ["deepseek-chat", "deepseek-reasoner"]
            return []

    def generate_text(self, prompt: str, model: str, **kwargs) -> str:
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get("options", {}).get("temperature", 0.7),
            )
            content = getattr(response.choices[0].message, "content", "") or ""
            # Only `content` is used (never reasoning_content); strip any inline thinking.
            return _strip_thinking(content)
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
            content = getattr(response.choices[0].message, "content", "") or ""
            content = _strip_thinking(content)
            # Extract the outermost JSON object so prose/thinking never breaks parsing.
            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end != -1 and end > start:
                content = content[start:end + 1]
            return json.loads(content)
        except Exception as e:
            self._handle_error(e, model)
            raise e

    def generate_structured(self, prompt: str, model: str, response_format: Type[BaseModel], **kwargs) -> BaseModel:
        # If running against third-party providers (NVIDIA, DeepSeek, OpenCode, OpenRouter, LM Studio),
        # use standard json_object mode and Pydantic validation directly.
        is_official_openai = self.client.base_url and "api.openai.com" in str(self.client.base_url)
        if not is_official_openai:
            json_data = self.generate_json(prompt, model, **kwargs)
            return response_format.model_validate(json_data)

        try:
            # Use beta parse if official OpenAI endpoint
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
            # If parse fails, fallback to JSON
            logger.warning(f"Generate structured parse failed, falling back to JSON mode: {e}")
            json_data = self.generate_json(prompt, model, **kwargs)
            return response_format.model_validate(json_data)

    def _handle_error(self, e, model):
         # Check for OpenRouter invalid model error (400)
        is_openrouter = self.client.base_url.host == "openrouter.ai" or "openrouter.ai" in str(self.client.base_url)
        if is_openrouter and "400" in str(e):
            logger.warning(f"OpenRouter model {model} failed. This might be due to model deprecation.")

    def generate_chat(self, messages: List[Dict[str, str]], model: str, **kwargs) -> LLMResult:
        """Native OpenAI-compatible chat: real roles (system/user/assistant).

        Works across all seven OpenAI-shaped providers by construction. Usage
        is captured from the completion envelope — the old layer discarded it.
        """
        started = time.monotonic()
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=kwargs.get("options", {}).get("temperature", 0.7),
            )
            content = getattr(response.choices[0].message, "content", "") or ""
            usage: Dict[str, int] = {}
            if getattr(response, "usage", None):
                u = response.usage
                usage = {
                    "prompt_tokens": getattr(u, "prompt_tokens", 0) or 0,
                    "completion_tokens": getattr(u, "completion_tokens", 0) or 0,
                    "total_tokens": getattr(u, "total_tokens", 0) or 0,
                }
            return LLMResult(
                content=_strip_thinking(content),
                provider="openai-compatible",
                model=model,
                latency_ms=int((time.monotonic() - started) * 1000),
                usage=usage,
            )
        except Exception as e:
            self._handle_error(e, model)
            raise classify_llm_error("openai-compatible", e) from e

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
            return _strip_thinking(response.text)
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
            content = _strip_thinking(response.text)
            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end != -1 and end > start:
                content = content[start:end + 1]
            return json.loads(content)
        except Exception as e:
            logger.error(f"Gemini JSON generation failed: {e}")
            raise e
            
    def generate_structured(self, prompt: str, model: str, response_format: Type[BaseModel], **kwargs) -> BaseModel:
         # Gemini SDK supports specific schema or just JSON. Fallback to JSON + Pydantic.
         # Future: Use `response_schema` in config if supported by pydantic mapping.
         json_data = self.generate_json(prompt, model, **kwargs)
         return response_format.model_validate(json_data)

    def generate_chat(self, messages: List[Dict[str, str]], model: str, **kwargs) -> LLMResult:
        """Native Gemini chat: system role → system_instruction; assistant→model.

        Usage captured from usage_metadata (prompt/candidates/total token counts).
        """
        started = time.monotonic()
        try:
            system_parts: List[str] = []
            contents: List[Dict[str, Any]] = []
            for m in messages:
                role = m.get("role", "user")
                text = m.get("content", "")
                if role == "system":
                    system_parts.append(text)
                else:
                    contents.append({
                        "role": "model" if role == "assistant" else "user",
                        "parts": [{"text": text}],
                    })
            if not contents:
                raise ValueError("Gemini requires at least one non-system message.")

            response = self.client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction="\n\n".join(system_parts) or None,
                    temperature=kwargs.get("options", {}).get("temperature", 0.7),
                ),
            )
            content = _strip_thinking(response.text or "")
            usage: Dict[str, int] = {}
            um = getattr(response, "usage_metadata", None)
            if um:
                usage = {
                    "prompt_tokens": getattr(um, "prompt_token_count", 0) or 0,
                    "completion_tokens": getattr(um, "candidates_token_count", 0) or 0,
                    "total_tokens": getattr(um, "total_token_count", 0) or 0,
                }
            return LLMResult(
                content=content,
                provider="gemini",
                model=model,
                latency_ms=int((time.monotonic() - started) * 1000),
                usage=usage,
            )
        except Exception as e:
            raise classify_llm_error("gemini", e) from e

# ---------------------------------------------------------------------------
# MiniMax Music 3 caption rewriter — official music-caption-rewriter port.
#
# Vendored static library (backend/data/caption-library): genre route table +
# 18 family indexes + ~1,000 caption templates from the official MiniMax Music 3
# repo. Workflow mirrors the official SKILL.md: route brief -> rank family ->
# score template filenames -> few-shot the real LLM -> synthesize a new
# three-heading caption. Never blocks generation: any failure degrades to a
# safe constructed caption.
# ---------------------------------------------------------------------------
_CAPTION_LIBRARY = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "caption-library",
)
_CAPTION_ROUTER_PATH = os.path.join(_CAPTION_LIBRARY, "references", "genre-router.md")
_CAPTION_INDEX_DIR = os.path.join(_CAPTION_LIBRARY, "references")
_CAPTION_TEMPLATE_DIR = os.path.join(_CAPTION_LIBRARY, "templates")


def _load_caption_family_routes() -> Dict[str, str]:
    """Parse genre-router.md into {family_slug: positive_cues}."""
    routes: Dict[str, str] = {}
    try:
        with open(_CAPTION_ROUTER_PATH, encoding="utf-8") as f:
            for line in f:
                m = re.match(r"\|\s*`([a-z0-9-]+)`\s*\|\s*([^|]+?)\s*\|\s*[^|]*\|\s*\[", line)
                if m:
                    routes[m.group(1)] = m.group(2).strip()
    except Exception as e:
        logger.warning(f"Caption router unavailable ({e}); rewriter uses fallback only.")
    return routes


def _list_caption_families() -> List[str]:
    """Family slugs from the vendored index-*.md files."""
    try:
        return sorted(
            name[len("index-"):-len(".md")]
            for name in os.listdir(_CAPTION_INDEX_DIR)
            if name.startswith("index-") and name.endswith(".md")
        )
    except Exception as e:
        logger.warning(f"Caption indexes unavailable ({e}).")
        return []


_CAPTION_FAMILY_ROUTES = _load_caption_family_routes()
_CAPTION_FAMILIES = _list_caption_families()


def _caption_tokens(text: Optional[str]) -> set:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def _rank_caption_families(query: str, limit: int = 2) -> List[str]:
    """Pick the 1-2 most relevant style families for a brief via token overlap."""
    q = _caption_tokens(query)
    scores = []
    for family in _CAPTION_FAMILIES:
        fam_tokens = set(family.split("-"))
        score = len(q & fam_tokens) * 3
        cues = _CAPTION_FAMILY_ROUTES.get(family, "")
        score += len(q & _caption_tokens(cues)) * 1
        if score:
            scores.append((score, family))
    scores.sort(key=lambda kv: -kv[0])
    return [f for _, f in scores[:limit]]


def _pick_caption_templates(query: str, families: List[str], k: int = 3) -> List[str]:
    """Score template filenames against brief tokens + routed families."""
    q = _caption_tokens(query)
    scored = []
    try:
        for name in os.listdir(_CAPTION_TEMPLATE_DIR):
            if not name.endswith(".txt"):
                continue
            stem = name[:-len(".txt")]
            score = len(q & _caption_tokens(stem))
            if any(stem.startswith(family) for family in families):
                score += 4
            if score:
                scored.append((score, name))
    except Exception as e:
        logger.warning(f"Caption template scan failed ({e}).")
        return []
    scored.sort(key=lambda kv: -kv[0])
    return [os.path.join(_CAPTION_TEMPLATE_DIR, n) for _, n in scored[:k]]


def _read_caption_file(path: str) -> str:
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


_CAPTION_REWRITE_SYSTEM = """You are a professional music caption writer for MiniMax Music 3.

Rewrite the user's brief into a structured caption with EXACTLY three sections, in this order:
1. Global Metadata — genre and subgenres, tempo, emotional progression (start -> peak -> resolve), application imagery, and sonic/production profile.
2. Vocal Details — for vocal music: lead vocal gender/timbre/register, delivery per section, harmony/backing vocals, and restrained vocal FX. For instrumental music: state explicitly that it is instrumental and name the instrument or texture carrying the lead melodic role.
3. Arrangement — a section-by-section timeline (Intro, Verse, Pre-Chorus, Chorus, Bridge, Outro): primary/secondary instrument lifecycles, groove development, transitions, embellishments, and spatial FX.

Rules:
- Treat bracketed lyric section tags as directives; keep them OUT of the caption and never merge them with lyric text.
- Never reproduce, paraphrase, or summarize lyric text.
- Do not copy sentences from the reference templates; synthesize a new caption around the brief.
- Use an exact BPM/key only when explicitly given; otherwise use ranges or qualitative tempo.
- Do not invent a precise key, BPM, vocal gender, or production technique beyond what the brief supports.
- Total length must be 250-450 words: Global Metadata about 120-180 words, Vocal Details about 70-110 words, Arrangement about 100-160 words. Prefer concrete musical changes over decorative prose.
- Respond ONLY with a JSON object: {"global_metadata": "...", "vocal_details": "...", "arrangement": "..."}"""


def _build_caption_rewrite_prompt(concept: str, lyrics: Optional[str], tags: Optional[str], templates: List[str]) -> str:
    brief = (concept or "").strip() or "A brand new song"
    parts = [f"USER BRIEF: {brief}"]
    if tags:
        parts.append(f"STYLE TAGS: {tags}")
    if lyrics:
        parts.append(f"LYRICS (bracketed tags are section directives; never put lyric text in the caption):\n{lyrics}")
    if templates:
        refs = []
        for t in templates:
            content = _read_caption_file(t)
            if content:
                refs.append(f"--- reference template: {os.path.basename(t)} ---\n{content}")
        if refs:
            parts.append("REFERENCE TEMPLATES (style guidance only — do not copy sentences):\n" + "\n\n".join(refs))
    return "\n\n".join(parts)


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
        
        if provider_name == "nvidia":
            api_key = config.get("nvidia", {}).get("api_key", "") or os.environ.get("NVIDIA_API_KEY", "")
            base_url = config.get("nvidia", {}).get("base_url", "https://integrate.api.nvidia.com/v1") or "https://integrate.api.nvidia.com/v1"
            return OpenAIProvider(
                api_key=api_key,
                base_url=base_url,
                timeout=45.0
            )
        elif provider_name == "ollama":
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
        elif provider_name == "opencode":
            api_key = config.get("opencode", {}).get("api_key", "") or os.environ.get("OPENCODE_API_KEY", "")
            base_url = config.get("opencode", {}).get("base_url", "https://opencode.ai/zen/go/v1")
            return OpenAIProvider(
                api_key=api_key,
                base_url=base_url
            )
        elif provider_name == "omlx":
            base_url = config.get("omlx", {}).get("base_url", "http://localhost:8787/v1")
            api_key = config.get("omlx", {}).get("api_key", "omlx")
            return OpenAIProvider(
                api_key=api_key or "omlx",
                base_url=base_url
            )
        else:
            return OllamaProvider(base_url="http://localhost:11434")

    @staticmethod
    def fetch_available_models(provider_name: str, api_key: Optional[str] = None, base_url: Optional[str] = None) -> List[str]:
        try:
            runtime_prov = ConfigManager().get_provider_config(provider_name)
            eff_key = str(api_key).strip() if api_key and str(api_key).strip() else runtime_prov.get("api_key", "")
            eff_url = str(base_url).strip() if base_url and str(base_url).strip() else runtime_prov.get("base_url", "")

            temp_config = {"provider": provider_name}
            if provider_name == "nvidia":
                temp_config["nvidia"] = {
                    "api_key": eff_key or os.environ.get("NVIDIA_API_KEY", ""),
                    "base_url": eff_url or "https://integrate.api.nvidia.com/v1"
                }
            elif provider_name == "ollama":
                temp_config["ollama"] = {"base_url": eff_url or "http://localhost:11434"}
            elif provider_name == "openai":
                temp_config["openai"] = {"api_key": eff_key}
            elif provider_name == "deepseek":
                temp_config["deepseek"] = {"api_key": eff_key, "base_url": eff_url or "https://api.deepseek.com"}
            elif provider_name == "openrouter":
                temp_config["openrouter"] = {"api_key": eff_key, "base_url": eff_url or "https://openrouter.ai/api/v1"}
            elif provider_name == "lmstudio":
                temp_config["lmstudio"] = {"base_url": eff_url or "http://localhost:1234/v1"}
            elif provider_name == "gemini":
                temp_config["gemini"] = {"api_key": eff_key}
            elif provider_name == "opencode":
                temp_config["opencode"] = {
                    "api_key": eff_key or os.environ.get("OPENCODE_API_KEY", ""),
                    "base_url": eff_url or "https://opencode.ai/zen/go/v1"
                }
            elif provider_name == "omlx":
                temp_config["omlx"] = {
                    "base_url": eff_url or "http://localhost:8787/v1",
                    "api_key": eff_key or "omlx"
                }
                
            provider = LLMService._get_provider(override_config=temp_config)
            return provider.get_models()
        except Exception as e:
            logger.error(f"Failed to fetch models for {provider_name}: {e}")
            raise e

    @staticmethod
    def get_models() -> List[str]:
        try:
            return LLMService._get_provider().get_models()
        except Exception as e:
            logger.warning(f"Failed to get models from active provider: {e}")
            return ["minimax-m3", "Llama-3.2-3B-Instruct-bf16", "llama3.2:3b-instruct-fp16"]

    @staticmethod
    def _get_active_model() -> str:
        config = ConfigManager().get_config()
        provider_name = config.get("provider", "nvidia")
        model = config.get(provider_name, {}).get("model")
        if model and str(model).strip():
            return str(model).strip()

        # Fallback only if no configured model exists
        try:
            provider = LLMService._get_provider()
            available = provider.get_models()
            if available:
                return available[0]
        except Exception as e:
            logger.warning(f"Failed to auto-detect model: {e}")

        return "deepseek-ai/deepseek-v4-flash-0731"

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
                f"Write complete, professional song lyrics about: {topic}.\n"
                "STRICT FORMAT REQUIREMENTS:\n"
                "1. Use standard section headers in brackets, each on its own line: [Intro], [Verse 1], [Chorus], [Verse 2], [Bridge], [Outro]\n"
                "2. Do NOT include any lyrics on the same line as a bracketed header.\n"
                "3. Output ONLY singable lyrics. NO conversational filler, NO explanations.\n"
                "4. NO stage directions or instrumental notations like '(guitar solo)', '(instrumental)', or '(repeat chorus)'."
            )
        
        response = provider.generate_text(prompt, model)
        from .lyrics_graph import sanitize_lyrics
        clean_response = sanitize_lyrics(response)
        
        try:
            with open("ai_debug.log", "a") as f:
                f.write(f"\n\n--- INITIAL GENERATION ({datetime.now().isoformat()}) ---\n")
                f.write(f"PROMPT:\n{prompt}\n")
                f.write(f"RESPONSE:\n{clean_response}\n")
        except Exception as e:
            print(f"Failed to write to debug log: {e}")
            
        return clean_response

    @staticmethod
    async def generate_lyrics_async(topic: str, model: Optional[str] = None, seed_lyrics: Optional[str] = None, tags: Optional[str] = None) -> str:
        """
        Async version of generate_lyrics that uses the pydantic-graph.
        Mode = CREATION
        """
        from .lyrics_graph import run_lyrics_graph, sanitize_lyrics
        
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
                return sanitize_lyrics(result["lyrics"])
            else:
                return sanitize_lyrics(seed_lyrics or "Generation failed.")
                
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
            from .lyrics_graph import sanitize_lyrics
            new_lyrics = sanitize_lyrics(engine.apply_edits(current_lyrics, result.operations))
            
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
        
        # Prepare provider attempts with automatic failover
        primary_name = ConfigManager().get_config().get("provider", "nvidia")
        providers_to_try = [(provider, model, primary_name)]
        
        full_cfg = ConfigManager().get_config()
        for candidate_name in ["nvidia", "deepseek", "opencode", "omlx", "ollama"]:
            if candidate_name != primary_name:
                c_cfg = full_cfg.get(candidate_name, {})
                c_key = c_cfg.get("api_key") or os.environ.get(f"{candidate_name.upper()}_API_KEY", "")
                if c_key or candidate_name in ["omlx", "ollama", "lmstudio"]:
                    try:
                        c_provider = LLMService._get_provider(override_config={"provider": candidate_name, candidate_name: c_cfg})
                        c_model = c_cfg.get("model") or "deepseek-ai/deepseek-v4-flash-0731"
                        providers_to_try.append((c_provider, c_model, candidate_name))
                    except Exception:
                        pass

        last_error = None
        for p_inst, p_model, p_name in providers_to_try:
            try:
                result = await run_lyrics_graph(
                    current_lyrics=current_lyrics,
                    user_message=user_message,
                    topic=topic,
                    tags=tags,
                    provider=p_inst,
                    model_name=p_model,
                )
                
                # Debug log success with FINAL LYRICS
                try:
                    with open("ai_debug.log", "a") as f:
                        f.write(f"--- GRAPH SUCCESS ({p_name}) ---\n")
                        f.write(f"AI Message: {result.get('message', 'N/A')}\n\n")
                        new_lyrics = result.get('lyrics', '')
                        f.write(f"--- NEW LYRICS ({len(new_lyrics)} chars) ---\n")
                        f.write(f"{new_lyrics}\n")
                        f.write(f"--- END NEW LYRICS ---\n")
                        f.write(f"{'='*60}\n\n")
                except:
                    pass
                
                if result.get("lyrics"):
                    return result
            except MaxRetriesExceededError as e:
                logger.warning(f"Lyrics graph max retries with {p_name}: {e}")
                last_error = e
            except Exception as e:
                logger.warning(f"Lyrics graph attempt failed with {p_name}: {e}")
                last_error = e

        return {
            "message": f"LLM lyric generation encountered an error: {str(last_error)}",
            "lyrics": current_lyrics,
            "error": True
        }

    @staticmethod
    def _extract_fallback_tags(concept: str) -> str:
        """Extract valid tags and subgenres from prompt keywords if LLM provider fails."""
        text = (concept or "").lower()
        matched: List[str] = []
        
        # Check against available registered styles
        all_styles = StyleRegistry().get_styles_for_prompt()
        for s in all_styles:
            if re.search(rf"\b{re.escape(s.lower())}\b", text):
                if s not in matched:
                    matched.append(s)

        # Keyword mapping for common production idioms
        kw_map = [
            (r"\b(hip[- ]?hop|rap|rapper|mc|bars|spit)\b", "Hip-Hop"),
            (r"\b(trap|808|drill)\b", "Trap"),
            (r"\b(gangster|gangsta|thug|street)\b", "Hip-Hop"),
            (r"\b(r[&n]b|rhythm and blues|soul|motown)\b", "R&B"),
            (r"\b(synthwave|retrowave|synth|80s)\b", "Synthesizer"),
            (r"\b(electronic|techno|house|edm|dance)\b", "Electronic"),
            (r"\b(pop|radio hit|anthem)\b", "Pop"),
            (r"\b(rock|guitar|punk|indie|grunge)\b", "Rock"),
            (r"\b(metal|heavy|hardcore)\b", "Heavy"),
            (r"\b(acoustic|unplugged|folk)\b", "Acoustic"),
            (r"\b(piano|ballad)\b", "Piano"),
            (r"\b(cinematic|orchestral|epic|film score)\b", "Cinematic"),
            (r"\b(lo[- ]?fi|chillhop|relax|mellow)\b", "Lofi"),
            (r"\b(female vocals?|woman|girl|singer)\b", "Female Vocal"),
            (r"\b(male vocals?|man|guy)\b", "Male Vocal"),
            (r"\b(dark|shadow|grim)\b", "Dark"),
            (r"\b(sad|melancholic|heartbreak)\b", "Emotional"),
        ]
        for pattern, tag in kw_map:
            if re.search(pattern, text, re.I):
                if tag not in matched:
                    matched.append(tag)

        if not matched:
            return "Pop, Modern DAW Master"
        return ", ".join(matched[:5])

    @staticmethod
    def _extract_fallback_title(concept: str) -> str:
        """Create a clean, creative song title from concept keywords if LLM is unavailable."""
        clean = re.sub(r'^(create|make|write|generate|produce|it is|it\'s|a|an|the|song about|track about)\s+', '', concept.strip(), flags=re.I)
        clean = re.sub(r'[^\w\s]', '', clean).strip()
        words = clean.split()
        if len(words) <= 4:
            return clean.title() if clean else "Studio Master"
        return " ".join(words[:4]).title()

    @staticmethod
    def _synthesize_fallback_lyrics(topic: str, tags: str) -> str:
        """Synthesize a complete structured 4-part song arrangement when upstream LLMs are unavailable."""
        return (
            f"[Verse 1]\n"
            f"Step inside the rhythm, hear the bass line start to roll\n"
            f"Every single melody is speaking to the soul\n"
            f"We started with a vision now we're taking to the stage\n"
            f"Writing brand new history across the open page\n\n"
            f"[Chorus]\n"
            f"Turn the sound up higher, feel the power and the flame\n"
            f"Nothing holds us under when we're mastering the game\n"
            f"From the ground into the skyline, we're the rhythm and the light\n"
            f"We own every heartbeat in the city through the night\n\n"
            f"[Verse 2]\n"
            f"Moving with momentum, every measure coming clear\n"
            f"Dropping in the pocket so the whole world wants to hear\n"
            f"Focused on the frequency, perfection in the mix\n"
            f"Nothing broken in the groove that passion cannot fix\n\n"
            f"[Outro]\n"
            f"Let the echo linger as the faders slowly fall\n"
            f"Standing at the pinnacle, we answered to the call"
        )

    @staticmethod
    def generate_title(context: str, model: Optional[str] = None) -> str:
        config = ConfigManager().get_config()
        primary_name = config.get("provider", "nvidia")
        providers_to_try = [primary_name, "nvidia", "deepseek", "omlx", "ollama"]
        
        prompt = f"Generate a short, creative, 2-5 word song title based on this concept/lyrics: '{context}'. Return ONLY the title, no quotes or prefix."
        
        for p_name in providers_to_try:
            try:
                p_cfg = config.get(p_name, {})
                p_inst = LLMService._get_provider(override_config={"provider": p_name, p_name: p_cfg})
                p_model = p_cfg.get("model") or model or "deepseek-ai/deepseek-v4-flash-0731"
                response = p_inst.generate_text(prompt, p_model).strip().replace('"', '').replace('\n', ' ')
                if response and len(response) < 60:
                    return response
            except Exception:
                pass
                
        return LLMService._extract_fallback_title(context)

    @staticmethod
    async def rewrite_caption(
        concept: str,
        lyrics: Optional[str] = None,
        tags: Optional[str] = None,
        model: Optional[str] = None,
        provider_config: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Rewrite a brief into a professional three-heading MiniMax structured caption.

        Uses the official music-caption-rewriter library (family routing + reference
        templates) with the real configured LLM provider. Production contract: this
        NEVER raises and NEVER blocks generation — any failure (LLM unreachable,
        unparseable output) degrades to a safe constructed caption with an honest
        fallback_reason so callers can surface it.
        """
        query = f"{concept or ''} {tags or ''}"
        families = _rank_caption_families(query)
        template_paths = _pick_caption_templates(query, families, k=3)
        prompt = _build_caption_rewrite_prompt(concept, lyrics, tags, template_paths)
        full_prompt = f"{_CAPTION_REWRITE_SYSTEM}\n\n{prompt}\n\nReturn the JSON object now."

        try:
            provider = LLMService._get_provider(provider_config)
            model = model or LLMService._get_active_model()
            result = provider.generate_json(full_prompt, model, options={"temperature": 0.7})
            caption = LLMService._parse_caption_response(result)
            if caption:
                return {
                    "structured_caption": caption,
                    "rewritten": True,
                    "fallback_reason": None,
                    "families": families,
                    "templates": [os.path.basename(t) for t in template_paths],
                }
            reason = "caption response did not contain all three sections"
        except Exception as e:
            reason = str(e)
            logger.warning(f"Caption rewrite failed ({reason}); using constructed fallback caption.")

        return {
            "structured_caption": LLMService._constructed_caption(concept, tags),
            "rewritten": False,
            "fallback_reason": reason,
            "families": families,
            "templates": [os.path.basename(t) for t in template_paths],
        }

    @staticmethod
    def _parse_caption_response(result: Any) -> Optional[Dict[str, str]]:
        """Validate an LLM caption response: all three sections, non-empty strings."""
        if not isinstance(result, dict):
            return None
        caption: Dict[str, str] = {}
        for key in ("global_metadata", "vocal_details", "arrangement"):
            value = result.get(key)
            if isinstance(value, str) and value.strip():
                caption[key] = value.strip()
        if len(caption) == 3:
            return caption
        return None

    @staticmethod
    def _constructed_caption(concept: Optional[str], tags: Optional[str]) -> Dict[str, str]:
        """Deterministic fallback caption (mirrors the provider's constructed path)."""
        tag_list = [t.strip() for t in (tags or "").split(",") if t.strip()]
        genre = tag_list[0] if tag_list else "Contemporary"
        tempo = tag_list[1] if len(tag_list) > 1 else "energetic"
        instruments = ", ".join(tag_list[2:]) if len(tag_list) > 2 else "Drums, Bass, Synths, Vocals"
        imagery = (concept or "").strip() or "A scene the song belongs to."
        return {
            "global_metadata": (
                f"Basic Attributes: Genre {genre}, tempo {tempo}.\n"
                f"Global Emotional Progression: Opens with the {tempo} {genre} character and builds in energy toward the chorus before resolving cleanly.\n"
                f"Application Scenarios & Imagery: {imagery}\n"
                f"Sonics & Production Profile: Polished, well-balanced mix with centered vocals and moderate stereo width."
            ),
            "vocal_details": (
                "Vocal Gender & Timbre: Singer A (Female), a clear and expressive vocal with strong presence.\n"
                "Vocal Style: Melodic and emotive throughout, with dynamic phrasing and a fuller delivery in the chorus.\n"
                "Harmony/Backing Vocals: Subtle stacked harmonies in the chorus.\n"
                "Vocal FX: Light reverb and delay for space without losing presence."
            ),
            "arrangement": (
                f"Instrument Lifecycle (Primary/Secondary): Primary {genre} foundation anchored by {instruments}.\n"
                f"Groove & Foundation Progression: Rhythmic drive throughout, thickening in the chorus and stripping back in the bridge.\n"
                f"Embellishments, Textures & Spatial FX: Moderate reverb tails and subtle risers on transitions."
            ),
        }

    @staticmethod
    def enhance_prompt(concept: str, model: Optional[str] = None) -> dict:
        config = ConfigManager().get_config()
        primary_name = config.get("provider", "nvidia")
        providers_to_try = [primary_name]
        for c in ["nvidia", "deepseek", "opencode", "openai", "gemini", "openrouter"]:
            if c not in providers_to_try:
                c_key = config.get(c, {}).get("api_key") or os.environ.get(f"{c.upper()}_API_KEY")
                if c_key:
                    providers_to_try.append(c)

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
            '{"topic": "A high-energy hip hop anthem with hard 808s and confident delivery.", "tags": "Hip-Hop, Trap, Male Vocal, Female Vocal"}'
        )

        for p_name in providers_to_try:
            try:
                p_cfg = config.get(p_name, {})
                p_inst = LLMService._get_provider(override_config={"provider": p_name, p_name: p_cfg})
                p_model = p_cfg.get("model") or model or "deepseek-ai/deepseek-v4-flash-0731"
                result = p_inst.generate_json(prompt, p_model)
                if isinstance(result, dict) and result.get("topic") and result.get("tags"):
                    return result
            except Exception as e:
                logger.warning(f"Enhance prompt failed with provider '{p_name}': {e}")

        # Intelligent Fallback extraction from prompt keywords
        fallback_tags = LLMService._extract_fallback_tags(concept)
        return {"topic": concept, "tags": fallback_tags}

    @staticmethod
    async def produce_full_track(concept: str, model: Optional[str] = None) -> dict:
        """Full-scale AI Producer synthesis:
        Derives topic, verified tags, title, complete structured lyrics, and structured captions.
        Uses a unified single-turn producer schema for fast, resilient response times.
        """
        clean_concept = (concept or "").strip()
        is_instrumental = bool(re.search(r'\b(instrumental|beat|backing track|lofi beat|no vocals?|karaoke track|ambient track)\b', clean_concept, re.I))

        valid_tags = StyleRegistry().get_styles_for_prompt()
        valid_tags_str = ", ".join(valid_tags)
        fallback_tags = LLMService._extract_fallback_tags(clean_concept)
        fallback_title = LLMService._extract_fallback_title(clean_concept)

        prompt = (
            f"Act as a professional music producer and songwriter. Transform this concept into a complete song specification.\n"
            f"USER CONCEPT: '{clean_concept}'\n\n"
            "INSTRUCTIONS:\n"
            "1. 'title': A short, creative 2-5 word song title.\n"
            "2. 'topic': A vivid 1-sentence musical direction.\n"
            f"3. 'tags': 3-5 style tags ONLY from this list: [{valid_tags_str}].\n"
            + ("4. 'lyrics': Empty string for instrumental." if is_instrumental else "4. 'lyrics': Complete singable lyrics formatted with standard section headers in brackets on their own lines: [Intro], [Verse 1], [Chorus], [Verse 2], [Bridge], [Outro]. No conversational filler, no stage directions.") + "\n\n"
            "Return ONLY a raw JSON object with keys 'title', 'topic', 'tags', 'lyrics'. Do NOT wrap in markdown."
        )

        title = fallback_title
        topic = clean_concept
        tags = fallback_tags
        lyrics = ""

        try:
            provider = LLMService._get_provider()
            active_model = model or LLMService._get_active_model()
            res = await asyncio.to_thread(provider.generate_json, prompt, active_model, options={"temperature": 0.7})
            if isinstance(res, dict):
                if res.get("title"):
                    title = str(res["title"]).strip().replace('"', '')
                if res.get("topic"):
                    topic = str(res["topic"]).strip()
                if res.get("tags"):
                    tags = str(res["tags"]).strip()
                if res.get("lyrics") and not is_instrumental:
                    lyrics = str(res["lyrics"]).strip()
        except Exception as e:
            logger.warning(f"Unified producer synthesis failed ({e}); using intelligent musical fallbacks.")

        if not is_instrumental and (not lyrics or len(lyrics) < 30):
            lyrics = LLMService._synthesize_fallback_lyrics(topic, tags)
        lyrics = _strip_thinking(lyrics)

        # Structured Caption via the official rewriter (three-heading contract)
        caption_result = await LLMService.rewrite_caption(
            concept=clean_concept,
            lyrics=lyrics or None,
            tags=tags,
            model=model,
        )
        structured_caption = caption_result.get("structured_caption", {})

        return {
            "title": title,
            "topic": topic,
            "tags": tags,
            "lyrics": lyrics,
            "structured_caption": structured_caption,
            "is_instrumental": is_instrumental,
            "llm_model": model or LLMService._get_active_model()
        }

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
        return ConfigManager().get_client_config()
