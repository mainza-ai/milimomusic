import requests
import logging
from typing import List, Optional
import json

logger = logging.getLogger(__name__)

import random

MUSIC_STYLES_LIBRARY = [
    "Cinematic", "Lo-fi", "Synthwave", "Rock", "HipHop", "Orchestral", "Ambient", "Trap", "Techno",
    "Jazz", "Blues", "Country", "Folk", "Reggae", "Soul", "R&B", "Funk", "Disco", "House", "Trance",
    "Dubstep", "Drum & Bass", "Jungle", "Garage", "Grime", "Afrobeats", "K-Pop", "J-Pop", "Indie Pop",
    "Dream Pop", "Shoegaze", "Post-Rock", "Math Rock", "Prog Rock", "Metal", "Punk", "Emo", "Grunge",
    "Acoustic", "Piano", "Classical", "Opera", "Gregorian Chant", "Medieval", "Celtic", "Nordic Folk",
    "Latin", "Salsa", "Bossa Nova", "Reggaeton", "Flamenco", "Tango", "Bollywood", "Indian Classical",
    "Gospel", "Spiritual", "Meditative", "New Age", "Dark Ambient", "Drone", "Noise", "Industrial",
    "Cyberpunk", "Vaporwave", "Chiptune", "Glitch", "IDM", "Complextro", "Electro Swing", "Nu-Disco",
    "Future Bass", "Tropical House", "Deep House", "Tech House", "Acid House", "Psytrance", "Hardstyle",
    "Breakbeat", "Trip-Hop", "Downtempo", "Chillout", "Lounge", "Elevator Music", "Muzak", "Experimental",
    "Avant-Garde", "Musique Concrete", "Minimalism", "Baroque", "Renaissance", "Romantic", "Impressionist"
]

class LLMService:
    BASE_URL = "http://localhost:11434"

    @staticmethod
    def get_models() -> List[str]:
        try:
            resp = requests.get(f"{LLMService.BASE_URL}/api/tags", timeout=2)
            if resp.status_code == 200:
                data = resp.json()
                return [model["name"] for model in data.get("models", [])]
        except Exception as e:
            logger.warning(f"Failed to fetch Ollama models: {e}")
        return []

    @staticmethod
    def generate_lyrics(topic: str, model: str = "llama3", seed_lyrics: Optional[str] = None) -> str:
        
        if seed_lyrics and seed_lyrics.strip():
            # EXPERIMENTAL: Expansion Prompt
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
            # Standard Generation
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
        
        try:
            resp = requests.post(
                f"{LLMService.BASE_URL}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=30
            )
            if resp.status_code == 200:
                return resp.json().get("response", "")
            else:
                raise Exception(f"Ollama Error: {resp.text}")
        except Exception as e:
            logger.error(f"Lyrics generation failed: {e}")
            raise e

    @staticmethod
    def generate_title(context: str, model: str = "llama3") -> str:
        prompt = f"Generate a short, creative, 2-5 word song title based on this concept/lyrics: '{context}'. Return ONLY the title, no quotes or prefix."
        
        try:
            resp = requests.post(
                f"{LLMService.BASE_URL}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=30
            )
            if resp.status_code == 200:
                return resp.json().get("response", "").strip().replace('"', '')
            else:
                logger.error(f"LLM Auto-Title Error: {resp.status_code} - {resp.text}")
                return "Untitled Track"
        except Exception as e:
            logger.error(f"LLM Auto-Title Exception: {e}")
            return "Untitled Track"
            
    @staticmethod
    def enhance_prompt(concept: str, model: str = "llama3") -> dict:
        """
        Takes a simple user concept (e.g. "sad song") and returns a rich JSON object
        with detailed topic description and style tags.
        """
        prompt = (
            f"Act as a professional music producer. Transform this simple user concept into a detailed musical direction.\n"
            f"USER CONCEPT: '{concept}'\n\n"
            "INSTRUCTIONS:\n"
            "1. Create a 'topic' description that is evocative and detailed (1 sentence).\n"
            "2. Select 3-5 'tags' that describe the genre, mood, instruments, and tempo (comma separated).\n"
            "3. Return ONLY a raw JSON object with keys 'topic' and 'tags'. Do NOT wrap in markdown code blocks.\n\n"
            "Example Output:\n"
            '{"topic": "A melancholic acoustic ballad about lost love in autumn.", "tags": "Acoustic, Folk, Sad, Guitar, Slow"}'
        )

        try:
            resp = requests.post(
                f"{LLMService.BASE_URL}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json" # Ollama supports JSON mode if model supports it, but we'll parse manually too
                },
                timeout=60
            )
            if resp.status_code == 200:
                raw_response = resp.json().get("response", "")
                # Clean response just in case
                raw_response = raw_response.strip()
                if raw_response.startswith("```json"):
                    raw_response = raw_response.replace("```json", "").replace("```", "")
                
                try:
                    return json.loads(raw_response)
                except json.JSONDecodeError:
                    # Fallback if LLM fails to json
                    logger.warning(f"LLM failed JSON format: {raw_response}")
                    return {"topic": concept, "tags": "Pop, Experimental"}
            else:
                raise Exception(f"Ollama Error: {resp.text}")
        except Exception as e:
            logger.error(f"Prompt enhancement failed: {e}")
            raise e

    @staticmethod
    def generate_inspiration(model: str = "llama3") -> dict:
        """
        Generates a random, creative song concept and style.
        """
        # High creativity prompt
        prompt = (
            "Act as a professional music producer brainstorming new hit songs.\n"
            "INSTRUCTIONS:\n"
            "1. Invent a UNIQUE, creative song concept/topic (1 vivid sentence).\n"
            "2. Select a matching musical style (3-5 tags like genre, mood, instruments).\n"
            "3. Return ONLY a raw JSON object with keys 'topic' and 'tags'.\n\n"
            "Examples:\n"
            '{"topic": "A lonely astronaut drifting through the cosmos.", "tags": "Ambient, Space, Ethereal"}\n'
            '{"topic": "A cyberpunk detective chasing a suspect in rain.", "tags": "Synthwave, Dark, Retro"}'
        )

        try:
            resp = requests.post(
                f"{LLMService.BASE_URL}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                    "options": {
                        "temperature": 0.9 # High creativity
                    }
                },
                timeout=60
            )
            if resp.status_code == 200:
                raw_response = resp.json().get("response", "")
                # Clean response just in case
                raw_response = raw_response.strip()
                if raw_response.startswith("```json"):
                    raw_response = raw_response.replace("```json", "").replace("```", "")
                
                try:
                    return json.loads(raw_response)
                except json.JSONDecodeError:
                    logger.warning(f"LLM failed JSON format: {raw_response}")
                    return {"topic": "A mysterious journey through time", "tags": "Orchestral, Epic, Cinematic"}
            else:
                raise Exception(f"Ollama Error: {resp.text}")
        except Exception as e:
            logger.error(f"Inspiration generation failed: {e}")
            raise e

    @staticmethod
    def generate_styles_list(model: str = "llama3") -> List[str]:
        """
        Generates a list of diverse music genres/styles using a static library for instant results.
        Returns 12 random styles.
        """
        try:
            return random.sample(MUSIC_STYLES_LIBRARY, 12)
        except Exception as e:
            logger.error(f"Style generation failed: {e}")
            # Fallback to a small slice if something goes wrong (unlikely)
            return MUSIC_STYLES_LIBRARY[:12]
