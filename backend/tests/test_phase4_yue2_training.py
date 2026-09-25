"""
Unit and integration tests for Phase 4: YuE2 48kHz Stereo Engine & 'My Music' Training Studio.
"""

import os
import wave
import pytest

from app.providers.yue2_provider import YuE2Provider
from app.services.training.yue2_trainer import (
    TrainingWorkflowMode,
    YuE2TrainingJob,
    YuE2TrainingStudio,
)


@pytest.mark.asyncio
async def test_yue2_provider_capabilities_and_generation():
    provider = YuE2Provider()
    caps = provider.get_capabilities()
    assert caps.provider_id == "yue2"
    assert caps.default_sample_rate == 48000
    assert caps.supports_lora is True

    # Test generation
    res = await provider.generate(
        job_id="test_yue2_gen_01",
        prompt="Electric synth rock anthem",
        lyrics="[Verse]\nNeon skies rising high",
        duration_ms=4000,
        tags="synth rock, energetic",
    )
    assert res.sample_rate == 48000
    assert os.path.exists(res.audio_path)

    # Verify output audio file is valid 48kHz stereo
    with wave.open(res.audio_path, "rb") as wf:
        assert wf.getframerate() == 48000
        assert wf.getnchannels() == 2
        assert wf.getsampwidth() == 2


def test_mothersuperior_instrumental_lora_auto_routing():
    provider = YuE2Provider()

    # Case 1: Instrumental track detected via tags
    loras_inst, is_inst = provider.resolve_lora_routing(
        tags="instrumental, ambient guitar",
        lyrics=None,
        active_loras=[{"path": "artist_vocal_lora.safetensors", "weight": 0.8}],
    )
    assert is_inst is True
    assert len(loras_inst) == 1
    assert "mothersuperior_instrumental_ar" in loras_inst[0]["path"]
    assert loras_inst[0]["weight"] == 1.0

    # Case 2: Vocal track
    loras_vocal, is_inst2 = provider.resolve_lora_routing(
        tags="pop ballad",
        lyrics="Hello from the stage",
        active_loras=[{"path": "my_vocal_style.safetensors", "weight": 0.7}],
    )
    assert is_inst2 is False
    assert len(loras_vocal) == 1
    assert loras_vocal[0]["path"] == "my_vocal_style.safetensors"


@pytest.mark.asyncio
async def test_yue2_training_auto_mode():
    job_id = "test_auto_job_99"
    job = YuE2TrainingStudio.create_job(
        job_id=job_id,
        dataset_name="my_rock_band",
        mode=TrainingWorkflowMode.AUTO,
    )
    assert job.mode == "auto"
    assert job.current_step == 0

    # Run auto pipeline
    completed_job = await YuE2TrainingStudio.run_auto_pipeline(job_id)
    assert completed_job.status == "completed"
    assert completed_job.current_step == 300
    assert "before" in completed_job.reconstruction_auditions
    assert 100 in completed_job.test_song_auditions
    assert completed_job.output_lora_path is not None


@pytest.mark.asyncio
async def test_yue2_training_guided_mode_stages():
    job_id = "test_guided_job_77"
    job = YuE2TrainingStudio.create_job(
        job_id=job_id,
        dataset_name="acoustic_sessions",
        mode=TrainingWorkflowMode.GUIDED,
        rank=64,
    )
    assert job.mode == "guided"
    assert job.rank == 64

    # Stage 1
    j1 = await YuE2TrainingStudio.run_guided_stage(job_id, stage=1)
    assert j1.status == "stage_1_ready_for_review"

    # Stage 2 (Acoustic adaptation with auditory before/after)
    j2 = await YuE2TrainingStudio.run_guided_stage(job_id, stage=2)
    assert j2.status == "stage_2_audition_ready"
    assert "original" in j2.reconstruction_auditions
    assert "after" in j2.reconstruction_auditions

    # Stage 3
    j3 = await YuE2TrainingStudio.run_guided_stage(job_id, stage=3)
    assert j3.status == "stage_3_complete"

    # Stage 4
    j4 = await YuE2TrainingStudio.run_guided_stage(job_id, stage=4)
    assert j4.status == "completed"
    assert j4.output_lora_path is not None
