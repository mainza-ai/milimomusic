"""Tests for video pipeline generators, lip sync providers, and director mechanics."""

import os
import sys
import tempfile
import pytest
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.video.types import SceneClip, SceneType, VideoPlan
from app.services.video.lip_sync.fallback import SmoothVisemeFallbackProvider
from app.services.video.lip_sync.cloud_lipsync import CloudLipSyncProvider
from app.services.video.generators.procedural import ProceduralVideoGenerator
from app.services.video.generators.cloud_video import CloudVideoGenerator
from app.services.video.video_director import VideoDirector


@pytest.fixture
def dummy_face_image(tmp_path):
    img_path = str(tmp_path / "dummy_face.png")
    im = Image.new("RGB", (256, 256), color=(200, 160, 140))
    im.save(img_path)
    return img_path


@pytest.fixture
def dummy_audio_file(tmp_path):
    # Generate a brief 1-second silent WAV file
    import wave
    import struct
    audio_path = str(tmp_path / "dummy_vocals.wav")
    with wave.open(audio_path, "w") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        # 16000 samples of low amplitude noise
        data = struct.pack("<16000h", *[100 if i % 2 == 0 else -100 for i in range(16000)])
        wav_file.writeframes(data)
    return audio_path


@pytest.mark.asyncio
async def test_smooth_viseme_fallback_provider(dummy_face_image, dummy_audio_file, tmp_path):
    """Test SmoothVisemeFallbackProvider generates valid animated video clip."""
    provider = SmoothVisemeFallbackProvider()
    out_video = str(tmp_path / "fallback_lip_sync.mp4")

    success = await provider.render_lip_sync(
        face_image_path=dummy_face_image,
        vocal_audio_path=dummy_audio_file,
        start_time=0.0,
        duration=1.0,
        out_path=out_video,
        width=320,
        height=320
    )
    assert success is True
    assert os.path.isfile(out_video)
    assert os.path.getsize(out_video) > 1000


@pytest.mark.asyncio
async def test_procedural_video_generator(tmp_path):
    """Test ProceduralVideoGenerator synthesizes valid dynamic video clip."""
    gen = ProceduralVideoGenerator()
    out_clip = str(tmp_path / "procedural_clip.mp4")

    success = await gen.generate_clip(
        prompt="Futuristic cityscape",
        duration=1.0,
        out_path=out_clip,
        width=320,
        height=240,
        visual_style="neon-cyberpunk"
    )
    assert success is True
    assert os.path.isfile(out_clip)
    assert os.path.getsize(out_clip) > 1000


def test_cloud_providers_availability():
    """Verify cloud providers check API keys properly."""
    cloud_lipsync = CloudLipSyncProvider(service="fal")
    if "FAL_KEY" not in os.environ:
        assert cloud_lipsync.is_available is False

    cloud_video = CloudVideoGenerator(service="replicate")
    if "REPLICATE_API_TOKEN" not in os.environ:
        assert cloud_video.is_available is False


def test_video_director_model_max_durations():
    """Verify VideoDirector returns correct max duration per model family."""
    assert VideoDirector.get_model_max_duration("wan_14b") == 5.0
    assert VideoDirector.get_model_max_duration("wan_1.3b") == 5.0
    assert VideoDirector.get_model_max_duration("ltx_video") == 10.0
    assert VideoDirector.get_model_max_duration("cogvideox") == 10.0
    assert VideoDirector.get_model_max_duration("hailuo_h3") == 15.0
