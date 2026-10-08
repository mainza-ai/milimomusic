import pytest
from app.services.llm_service import LLMService
from app.services.video.video_director import VideoDirector


def test_resolve_model_max_tokens():
    # OpenAI reasoning models
    assert LLMService.resolve_model_max_tokens("o1-preview") == 65536
    assert LLMService.resolve_model_max_tokens("o3-mini") == 65536

    # GPT-4o
    assert LLMService.resolve_model_max_tokens("gpt-4o") == 16384

    # Claude 3.5 / 3.7
    assert LLMService.resolve_model_max_tokens("claude-3-7-sonnet") == 16384
    assert LLMService.resolve_model_max_tokens("claude-3-5-sonnet") == 8192

    # DeepSeek
    assert LLMService.resolve_model_max_tokens("deepseek-chat") == 8192

    # Local models & local providers
    assert LLMService.resolve_model_max_tokens("Qwen3.8-27B-Uncensored-OrcaRouter-MLX-8bit", "omlx") == 16384
    assert LLMService.resolve_model_max_tokens("Llama-3.2-3B-Instruct-bf16", "omlx") == 16384
    assert LLMService.resolve_model_max_tokens("gemma-4-12b", "lmstudio") == 16384

    # Default fallback
    assert LLMService.resolve_model_max_tokens("unknown-model", "unknown-provider") == 8192


def test_extract_json_treatment_truncated_recovery():
    director = VideoDirector()
    
    truncated_text = """
    {
      "concept_title": "SWITCH",
      "logline": "High energy drill video",
      "scenes": [
        {
          "clip_index": 1,
          "scene_type": "VOCAL_PERFORMANCE",
          "visual_action": "Rapper leans into the camera lens with intense swagger",
          "camera_motion": "Low-angle push-in",
          "lighting_and_atmosphere": "Cyan rim lighting"
        },
        {
          "clip_index": 2,
          "scene_type": "NARRATIVE_STORY",
          "visual_action": "Armored vehicle drifts across wet asphalt",
          "camera_motion": "High-speed tracking shot",
          "lighting_and_atmosphere": "Red taillight glow"
        },
        {
          "clip_index": 3,
          "scene_type": "VOCAL_PERFORMANCE",
          "visual_action": "Protagonist adjusts his tactical vest
    """

    parsed = director._extract_json_treatment(truncated_text)
    assert parsed is not None
    assert "scenes" in parsed
    assert len(parsed["scenes"]) == 2
    assert parsed["scenes"][0]["clip_index"] == 1
    assert parsed["scenes"][1]["clip_index"] == 2
    assert parsed["scenes"][0]["visual_action"] == "Rapper leans into the camera lens with intense swagger"


def test_extract_json_treatment_array_unwrapping():
    director = VideoDirector()

    # Model returned a raw JSON array of scenes instead of an object
    array_text = """
    [
      {
        "clip_index": 1,
        "visual_action": "Opening shot"
      },
      {
        "clip_index": 2,
        "visual_action": "Second shot"
      }
    ]
    """
    parsed = director._extract_json_treatment(array_text)
    assert parsed is not None
    assert "scenes" in parsed
    assert len(parsed["scenes"]) == 2


def test_extract_json_treatment_nested_treatment_wrapper():
    director = VideoDirector()

    # Model wrapped under "treatment": {"scenes": [...]}
    wrapped_text = """
    {
      "treatment": {
        "concept_title": "Wrapped Title",
        "scenes": [
          {"clip_index": 1, "visual_action": "Scene 1"}
        ]
      }
    }
    """
    parsed = director._extract_json_treatment(wrapped_text)
    assert parsed is not None
    assert "scenes" in parsed
    assert parsed["concept_title"] == "Wrapped Title"
    assert len(parsed["scenes"]) == 1
