"""Tests for video_service and video API endpoints."""

import os
import sys
import uuid
from pathlib import Path
from unittest.mock import patch, AsyncMock

import pytest
import httpx
from sqlmodel import Session

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.main import app, engine
from app.models import Job, JobStatus
from app.services.video_service import video_service


@pytest.fixture()
def client():
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://testserver")


@pytest.fixture()
def sample_job():
    job_id = uuid.uuid4()
    job = Job(
        id=job_id,
        title="Cyber Drift",
        prompt="Synthesizer synthwave high energy electronic beat",
        lyrics="[Verse 1]\nDriving through the neon night\nElectric reflections in my sight",
        status=JobStatus.COMPLETED,
        duration_ms=60000,
        audio_path="generated_audio/test_cyber_drift.mp3",
    )
    with Session(engine) as session:
        session.add(job)
        session.commit()
        session.refresh(job)

    yield job

    with Session(engine) as session:
        existing = session.get(Job, job_id)
        if existing:
            session.delete(existing)
            session.commit()


@pytest.mark.asyncio
async def test_generate_storyboard_scenes(sample_job):
    """Test storyboard scene generation produces required camera, lighting, and timeline fields."""
    scenes = await video_service.generate_storyboard(sample_job, visual_style="neon-cyberpunk")
    assert isinstance(scenes, list)
    assert len(scenes) >= 3

    for scene in scenes:
        assert "time" in scene
        assert "prompt" in scene
        assert "camera" in scene
        assert "lighting" in scene
        assert len(scene["prompt"]) > 0


@pytest.mark.asyncio
async def test_generate_storyboard_style_customization(sample_job):
    """Test storyboard generation customizes lighting and aesthetics per visual style."""
    retro_scenes = await video_service.generate_storyboard(sample_job, visual_style="retro-vhs")
    assert any("retro" in s["prompt"].lower() or "vhs" in s["lighting"].lower() or "analog" in s["prompt"].lower() for s in retro_scenes)


@pytest.mark.asyncio
async def test_storyboard_api_endpoint(client, sample_job):
    """Test POST /videos/storyboard/{job_id} endpoint returns 200 with scenes."""
    response = await client.post(
        f"/videos/storyboard/{sample_job.id}",
        json={"visual_style": "anime-cinematic"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["job_id"] == str(sample_job.id)
    assert data["visual_style"] == "anime-cinematic"
    assert len(data["scenes"]) >= 3


@pytest.mark.asyncio
async def test_storyboard_api_endpoint_404(client):
    """Test POST /videos/storyboard/{job_id} returns 404 for non-existent job."""
    bad_id = str(uuid.uuid4())
    response = await client.post(f"/videos/storyboard/{bad_id}", json={})
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_render_audio_reactive_video_execution(sample_job, tmp_path):
    """Test render_audio_reactive_video handles audio resolution and mock ffmpeg execution."""
    audio_dir = Path("generated_audio")
    audio_dir.mkdir(parents=True, exist_ok=True)
    audio_file = audio_dir / "test_cyber_drift.mp3"
    audio_file.write_bytes(b"\x00" * 1024)

    try:
        with patch("asyncio.create_subprocess_exec") as mock_proc:
            mock_process = AsyncMock()
            mock_process.returncode = 0
            mock_process.communicate.return_value = (b"", b"")
            mock_proc.return_value = mock_process

            video_url = await video_service.render_audio_reactive_video(
                sample_job,
                visual_style="neon-cyberpunk",
                resolution="720p"
            )
            assert video_url.startswith("/audio/videos/")
            assert video_url.endswith("_reactive.mp4")
    finally:
        if audio_file.exists():
            audio_file.unlink()


@pytest.mark.asyncio
async def test_render_endpoint_validation(client):
    """Test POST /videos/render/{job_id} properly rejects missing jobs and jobs with no audio."""
    # 404 on unknown job
    bad_id = str(uuid.uuid4())
    response = await client.post(f"/videos/render/{bad_id}", json={})
    assert response.status_code == 404

    # 400 on job without audio
    no_audio_id = uuid.uuid4()
    job = Job(id=no_audio_id, title="No Audio", prompt="test", status=JobStatus.QUEUED, audio_path=None)
    with Session(engine) as session:
        session.add(job)
        session.commit()

    try:
        response = await client.post(f"/videos/render/{no_audio_id}", json={})
        assert response.status_code == 400
        assert "no completed audio" in response.json()["detail"].lower()
    finally:
        with Session(engine) as session:
            existing = session.get(Job, no_audio_id)
            if existing:
                session.delete(existing)
                session.commit()


@pytest.mark.asyncio
async def test_get_video_providers_endpoint(client):
    """Test GET /videos/providers returns supported hardware and providers."""
    response = await client.get("/videos/providers")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 3
    provider_ids = [p["id"] for p in data]
    assert "local_wan_14b" in provider_ids
    assert "cloud_fal" in provider_ids
    assert "cloud_replicate" in provider_ids


@pytest.mark.asyncio
async def test_video_director_segment_song(sample_job):
    """Test VideoDirector segments song into musical beats with vocal/b-roll balance."""
    from app.services.video.video_director import video_director
    plan = video_director.segment_song(
        job=sample_job,
        max_clip_duration=5.0,
        bpm=120.0,
        visual_style="neon-cyberpunk",
        character_desc="Cyberpunk pop star"
    )
    assert plan.total_clips > 0
    assert len(plan.clips) == plan.total_clips
    assert plan.vocal_clips_count >= 1
    for clip in plan.clips:
        assert clip.duration > 0
        assert clip.prompt
        assert clip.camera
        assert clip.lighting


@pytest.mark.asyncio
async def test_plan_video_endpoint(client, sample_job):
    """Test POST /videos/plan/{job_id} generates production scene plan."""
    response = await client.post(
        f"/videos/plan/{sample_job.id}",
        json={
            "model_name": "wan_14b",
            "max_clip_duration": 5.0,
            "bpm": 120.0,
            "visual_style": "neon-cyberpunk"
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["job_id"] == str(sample_job.id)
    assert data["total_clips"] > 0
    assert "clips" in data
    assert len(data["clips"]) == data["total_clips"]


@pytest.mark.asyncio
async def test_generate_keyframes_endpoint(client, sample_job):
    """Test POST /videos/keyframes/{job_id} returns keyframe list."""
    with patch("app.services.image_service.image_service.generate_scene_background", return_value={"ok": False, "dest_path": None}):
        response = await client.post(
            f"/videos/keyframes/{sample_job.id}",
            json={
                "visual_style": "neon-cyberpunk",
                "resolution": "720p"
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "keyframes" in data
        assert len(data["keyframes"]) > 0
        assert "clip_index" in data["keyframes"][0]


@pytest.mark.asyncio
async def test_ass_karaoke_subtitle_generation(sample_job, tmp_path):
    """Test video_orchestrator creates valid Advanced SubStation Alpha script."""
    from app.services.video.video_orchestrator import video_orchestrator
    from app.services.video.video_director import video_director

    plan = video_director.segment_song(sample_job, max_clip_duration=5.0, bpm=120.0)
    timed_lines = [{"start": 0.0, "end": 4.0, "text": "Driving through the neon night"}]
    ass_content = video_orchestrator.generate_karaoke_ass(
        timed_lines=timed_lines,
        width=1280,
        height=720,
        style="neon-cyberpunk",
        subtitle_style="neon"
    )
    assert "[Script Info]" in ass_content
    assert "Format: Layer, Start, End, Style" in ass_content
    assert "Driving through the neon night" in ass_content


def test_resolve_engine_for_video_model():
    """Verify video model mapping precisely separates Wan 14B vs 1.3B, LTX, H3, and CogVideoX."""
    from app.services.video_service import resolve_engine_for_video_model

    assert resolve_engine_for_video_model({"id": "wan2_1_t2v_14b", "name": "Wan2.1 T2V (14B Open Flagship)"}) == "wan_14b"
    assert resolve_engine_for_video_model({"id": "wan2_1_t2v_1_3b", "name": "Wan2.1 T2V (1.3B Open Lightweight)"}) == "wan_1.3b"
    assert resolve_engine_for_video_model({"id": "minimax_h3", "name": "MiniMax Hailuo 3"}) == "hailuo_h3"
    assert resolve_engine_for_video_model({"id": "minimax_h3_mlx_8bit", "name": "MiniMax Hailuo 3 MLX"}) == "hailuo_h3"
    assert resolve_engine_for_video_model({"id": "cogvideox_5b", "name": "CogVideoX-1.5-5B"}) == "cogvideox"
    assert resolve_engine_for_video_model({"id": "hunyuan_video", "name": "Tencent HunyuanVideo"}) == "hunyuan"
    assert resolve_engine_for_video_model({"id": "custom_ltx", "repo_id": "Lightricks/LTX-Video", "name": "LTX-Video 0.9B"}) == "ltx_video"
    # Legacy alias normalization
    assert resolve_engine_for_video_model({"id": "wan2.1"}) == "wan_14b"


def test_get_available_video_models_registry():
    """Verify get_available_video_models returns canonical VideoModelKey entries matching frontend expectations."""
    from app.services.video_service import video_service

    models = video_service.get_available_video_models()
    expected_keys = {"wan_14b", "wan_1.3b", "ltx_video", "cogvideox", "hailuo_h3", "hunyuan", "audioreactive"}
    for key in expected_keys:
        assert key in models, f"Expected key '{key}' missing from get_available_video_models()"
        assert "max_duration" in models[key]
        assert "local_weights_present" in models[key]
        assert isinstance(models[key]["local_weights_present"], bool)


@pytest.mark.asyncio
async def test_active_video_engine_api_sync(client):
    """Test GET and POST /videos/active-engine bidirectional synchronization."""
    # 1. Read current active engine
    res = await client.get("/videos/active-engine")
    assert res.status_code == 200
    data = res.json()
    assert "engine" in data
    assert "model_id" in data

    # 2. Switch to wan_1.3b
    switch_res = await client.post("/videos/active-engine", json={"engine": "wan_1.3b"})
    assert switch_res.status_code == 200
    switch_data = switch_res.json()
    assert switch_data["status"] == "ok"
    assert switch_data["engine"] == "wan_1.3b"
    assert switch_data["model_id"] == "wan2_1_t2v_1_3b"

    # 3. Verify GET reflects new active engine
    verify_res = await client.get("/videos/active-engine")
    assert verify_res.status_code == 200
    assert verify_res.json()["engine"] == "wan_1.3b"

    # 4. Switch back to wan_14b
    restore_res = await client.post("/videos/active-engine", json={"engine": "wan_14b"})
    assert restore_res.status_code == 200
    assert restore_res.json()["engine"] == "wan_14b"
    assert restore_res.json()["model_id"] == "wan2_1_t2v_14b"


def test_video_generator_registry_and_model_specs():
    """Verify VideoGeneratorRegistry accurately resolves generators without silent Wan fallthroughs."""
    from app.services.video.generator_registry import VideoGeneratorRegistry
    from app.services.video.generators.minimax_h3 import MiniMaxH3Generator
    from app.services.video.generators.diffusers_wan import DiffusersWanGenerator
    from app.services.video.generators.diffusers_ltx import DiffusersLTXGenerator
    from app.services.video.model_specs import get_model_spec

    # 1. MiniMax Hailuo H3 resolution
    gen_hailuo = VideoGeneratorRegistry.resolve("hailuo_h3", "local")
    assert isinstance(gen_hailuo, MiniMaxH3Generator)
    assert gen_hailuo.name == "hailuo_h3"

    gen_minimax = VideoGeneratorRegistry.resolve("minimax_h3", "local")
    assert isinstance(gen_minimax, MiniMaxH3Generator)

    # 2. Wan variants resolution
    gen_wan14 = VideoGeneratorRegistry.resolve("wan_14b", "local")
    assert isinstance(gen_wan14, DiffusersWanGenerator)
    assert gen_wan14.model_size == "14b"

    gen_wan13 = VideoGeneratorRegistry.resolve("wan_1.3b", "local")
    assert isinstance(gen_wan13, DiffusersWanGenerator)
    assert gen_wan13.model_size == "1.3b"

    # 3. LTX-Video resolution
    gen_ltx = VideoGeneratorRegistry.resolve("ltx_video", "local")
    assert isinstance(gen_ltx, DiffusersLTXGenerator)

    # 4. Model specs contracts
    spec_h3 = get_model_spec("hailuo_h3")
    assert spec_h3.fps == 24
    assert spec_h3.max_duration == 15.0
    assert spec_h3.min_frames == 49
    assert spec_h3.frame_step == 48

    frames, dur = spec_h3.compute_lattice_frames(10.0)
    assert frames == 49 + 48 * 4  # 241 frames or valid lattice
    assert dur > 0

    spec_wan = get_model_spec("wan_14b")
    assert spec_wan.fps == 16
    assert spec_wan.max_duration == 5.0





def test_canonical_audio_and_stem_resolution_regression(tmp_path):
    """Regression test verifying resolve_audio_file and resolve_stem_file handle /audio/ paths and stems."""
    from app.core.paths import get_generated_audio_dir, resolve_audio_file, resolve_stem_file, resolve_image_file
    from app.services.video.video_orchestrator import video_orchestrator

    gen_dir = get_generated_audio_dir()
    test_uuid = uuid.uuid4()
    test_wav = gen_dir / f"{test_uuid}.wav"
    test_wav.write_bytes(b"\x00" * 2048)

    stems_dir = gen_dir / "stems"
    stems_dir.mkdir(parents=True, exist_ok=True)
    test_vocals = stems_dir / f"{test_uuid}_vocals.wav"
    test_vocals.write_bytes(b"\x00" * 2048)

    try:
        # Test URL-style path resolution
        resolved = resolve_audio_file(f"/audio/{test_uuid}.wav")
        assert resolved is not None
        assert os.path.isfile(resolved)
        assert os.path.basename(resolved) == f"{test_uuid}.wav"

        # Test video_orchestrator delegation
        orch_resolved = video_orchestrator.resolve_audio_path(f"/audio/{test_uuid}.wav")
        assert orch_resolved is not None
        assert orch_resolved == resolved

        # Test stem resolution from disk without stems_json
        stem_resolved = resolve_stem_file(test_uuid, "vocals")
        assert stem_resolved is not None
        assert os.path.isfile(stem_resolved)
        assert f"{test_uuid}_vocals.wav" in stem_resolved

        # Test stem resolution via mock Job
        mock_job = Job(id=test_uuid, title="Test Track", prompt="test", status=JobStatus.COMPLETED, audio_path=f"/audio/{test_uuid}.wav")
        vocal_path = video_orchestrator.resolve_vocals_stem(mock_job)
        assert vocal_path is not None
        assert os.path.isfile(vocal_path)

    finally:
        if test_wav.exists():
            test_wav.unlink()
        if test_vocals.exists():
            test_vocals.unlink()


@pytest.mark.asyncio
async def test_director_treatment_endpoints(client, sample_job):
    """Test generating and retrieving AI Visual Director treatment via REST API."""
    # 1. Generate treatment
    post_res = await client.post(
        f"/videos/director-treatment/{sample_job.id}",
        json={
            "visual_style": "film-noir-35mm",
            "model_name": "wan_14b",
            "pacing_bias": 1,
            "character_desc": "Detective in trench coat"
        }
    )
    assert post_res.status_code == 200
    data = post_res.json()
    assert data["status"] == "ok"
    assert "treatment" in data
    treatment = data["treatment"]
    assert "concept_title" in treatment
    assert "visual_metaphor" in treatment
    assert "character_profile" in treatment
    assert len(treatment["scenes"]) >= 3

    for s in treatment["scenes"]:
        assert "visual_action" in s
        assert "prompt" in s
        assert "camera" in s
        assert "lighting" in s
        assert "section_label" in s
        assert "musical_energy" in s

    # 2. Retrieve cached treatment
    get_res = await client.get(f"/videos/director-treatment/{sample_job.id}")
    assert get_res.status_code == 200
    cached = get_res.json()
    assert cached["status"] == "ok"
    assert cached["treatment"]["concept_title"] == treatment["concept_title"]


@pytest.mark.asyncio
async def test_reimagine_scene_endpoint(client, sample_job):
    """Test requesting AI Director to re-conceive an individual scene."""
    res = await client.post(
        f"/videos/director-treatment/{sample_job.id}/re-imagine-scene/2",
        json={
            "user_instruction": "Explosive slow-motion rain falling upward into neon clouds",
            "visual_style": "neon-cyberpunk",
            "current_scene": {
                "clip_index": 2,
                "scene_type": "METAPHORICAL_VISUAL",
                "musical_energy": 4
            }
        }
    )
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["clip_index"] == 2
    scene = data["scene"]
    assert "prompt" in scene
    assert "camera" in scene
    assert "lighting" in scene


@pytest.mark.asyncio
async def test_retake_clip_custom_prompt_keyframe(client, sample_job):
    """Test retake endpoint accepts custom prompt and returns keyframe url."""
    with patch("app.services.image_service.image_service.generate_scene_background", return_value={"ok": False, "dest_path": None}):
        res = await client.post(
            f"/videos/retake-clip/{sample_job.id}/1",
            json={
                "prompt": "An extreme close up of cyberpunk glasses reflecting laser grids",
                "visual_style": "neon-cyberpunk",
                "resolution": "720p"
            }
        )
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "ok"
        assert data["clip_index"] == 1
        assert "keyframe_url" in data
        assert data["prompt"] == "An extreme close up of cyberpunk glasses reflecting laser grids"


@pytest.mark.asyncio
async def test_keyframes_get_and_post_api_endpoints(client, sample_job):
    """Test generating keyframes via POST and retrieving them via GET /videos/keyframes/{job_id}."""
    with patch("app.services.image_service.image_service.generate_scene_background", return_value={"ok": False, "dest_path": None}):
        # 1. POST /videos/keyframes/{job_id}
        post_res = await client.post(
            f"/videos/keyframes/{sample_job.id}",
            json={
                "visual_style": "neon-cyberpunk",
                "resolution": "720p",
                "aspect_ratio": "16:9",
                "force_regenerate": True,
            }
        )
        assert post_res.status_code == 200
        data = post_res.json()
        assert data["status"] == "ok"
        assert "keyframes" in data
        assert len(data["keyframes"]) >= 3
        first_kf = data["keyframes"][0]
        assert first_kf["clip_index"] == 1
        assert "keyframe_url" in first_kf

        # 2. GET /videos/keyframes/{job_id}
        get_res = await client.get(f"/videos/keyframes/{sample_job.id}")
        assert get_res.status_code == 200
        get_data = get_res.json()
        assert get_data["status"] == "ok"
        assert get_data["job_id"] == str(sample_job.id)
        assert "keyframes" in get_data
        assert len(get_data["keyframes"]) >= 3
        assert 1 in get_data["keyframes"] or "1" in get_data["keyframes"]


@pytest.mark.asyncio
async def test_generate_scene_keyframes_unique_and_decoupled_from_cover(tmp_path):
    """Test that generate_scene_keyframes creates distinct images for each clip,
    including vocal clips, and never copies the album cover image.
    """
    import hashlib
    from PIL import Image
    from app.services.video.video_orchestrator import video_orchestrator

    # 1. Create a dummy cover image with a known color and MD5
    cover_path = tmp_path / "album_cover.png"
    img = Image.new("RGB", (512, 512), color=(255, 0, 0))
    img.save(cover_path)
    cover_md5 = hashlib.md5(cover_path.read_bytes()).hexdigest()

    # 2. Create mock job
    job = Job(
        id=uuid.uuid4(),
        title="Keyframe Test",
        prompt="A testing song",
        cover_image_path=str(cover_path),
        status=JobStatus.COMPLETED
    )

    # 3. Create 3 clips: Clip 1 (Vocal), Clip 2 (Narrative), Clip 3 (Vocal)
    user_scenes = [
        {
            "clip_index": 1,
            "start_time": 0.0,
            "end_time": 5.0,
            "duration": 5.0,
            "time_str": "0.0s - 5.0s",
            "is_vocal": True,
            "scene_type": "VOCAL_PERFORMANCE",
            "prompt": "Close-up of a vocalist singing with neon blue hair under volumetric laser beams",
        },
        {
            "clip_index": 2,
            "start_time": 5.0,
            "end_time": 10.0,
            "duration": 5.0,
            "time_str": "5.0s - 10.0s",
            "is_vocal": False,
            "scene_type": "NARRATIVE_STORY",
            "prompt": "A futuristic hovercar speeding down an empty rainy expressway at midnight",
        },
        {
            "clip_index": 3,
            "start_time": 10.0,
            "end_time": 15.0,
            "duration": 5.0,
            "time_str": "10.0s - 15.0s",
            "is_vocal": True,
            "scene_type": "VOCAL_PERFORMANCE",
            "prompt": "Medium shot of the lead singer on a high-rise balcony looking at dystopian skyline",
        },
    ]

    # 4. Generate keyframes
    with patch("app.services.image_service.image_service.generate_scene_background", return_value={"ok": False, "dest_path": None}):
        kf_results = await video_orchestrator.generate_scene_keyframes(
            job=job,
            visual_style="neon-cyberpunk",
            width=1280,
            height=720,
            force_regenerate=True,
            user_scenes=user_scenes
        )

    assert len(kf_results) == 3

    # 5. Verify every clip has a generated keyframe file on disk
    kf_hashes = []
    for kf in kf_results:
        kf_path = Path(kf["keyframe_path"])
        assert kf_path.exists(), f"Keyframe file {kf_path} does not exist"
        data = kf_path.read_bytes()
        h = hashlib.md5(data).hexdigest()
        assert h != cover_md5, f"Clip #{kf['clip_index']} duplicated the album cover image!"
        kf_hashes.append(h)

    # 6. Verify that keyframes for different clips are distinct (not duplicated across clips)
    assert len(set(kf_hashes)) == len(kf_results), "Duplicate keyframe stills were detected between clips!"


@pytest.mark.asyncio
async def test_clear_video_timeline_endpoint(client, sample_job):
    """Test DELETE /videos/timeline/{job_id} clears planned scenes and keyframes."""
    # Seed video_config_json with scenes and treatment
    import json
    with Session(engine) as session:
        j = session.get(Job, sample_job.id)
        j.video_config_json = json.dumps({
            "director_treatment": {"concept_title": "Test Title", "scenes": []},
            "scenes": [{"clip_index": 1, "prompt": "Test scene"}],
            "visual_style": "neon-cyberpunk"
        })
        session.add(j)
        session.commit()

    res = await client.delete(f"/videos/timeline/{sample_job.id}")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "cleared"
    assert data["job_id"] == str(sample_job.id)

    with Session(engine) as session:
        j = session.get(Job, sample_job.id)
        if j.video_config_json:
            cfg = json.loads(j.video_config_json)
            assert "director_treatment" not in cfg
            assert "scenes" not in cfg


@pytest.mark.asyncio
async def test_plan_music_video_treatment_and_force_refresh(client, sample_job):
    """Test POST /videos/plan/{job_id} returns treatment and persists it to video_config_json."""
    import json
    res = await client.post(
        f"/videos/plan/{sample_job.id}",
        json={
            "visual_style": "neon-cyberpunk",
            "model_name": "wan_14b",
            "use_llm": False,
            "force_refresh": True
        }
    )
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert "treatment" in data
    assert data["treatment"] is not None
    assert "concept_title" in data["treatment"]
    assert len(data["clips"]) > 0

    with Session(engine) as session:
        j = session.get(Job, sample_job.id)
        assert j.video_config_json is not None
        cfg = json.loads(j.video_config_json)
        assert "director_treatment" in cfg
        assert "scenes" in cfg


@pytest.mark.asyncio
async def test_plan_music_video_async_endpoint_and_polling(client, sample_job):
    """Test POST /videos/plan/{job_id} with async_mode=True returns queued task and completes with clips & treatment."""
    import asyncio
    res = await client.post(
        f"/videos/plan/{sample_job.id}",
        json={
            "visual_style": "neon-cyberpunk",
            "model_name": "wan_14b",
            "use_llm": False,
            "force_refresh": True,
            "async_mode": True
        }
    )
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "queued"
    assert "task_id" in data
    task_id = data["task_id"]
    assert task_id.startswith("plan_")

    # Poll status until completed
    for _ in range(30):
        poll_res = await client.get(f"/videos/tasks/{task_id}")
        assert poll_res.status_code == 200
        poll_data = poll_res.json()
        assert poll_data["id"] == task_id
        if poll_data["status"] == "completed":
            assert poll_data["progress"] == 100
            assert len(poll_data["clips"]) > 0
            assert poll_data["treatment"] is not None
            break
        await asyncio.sleep(0.2)
    else:
        pytest.fail(f"Async planning task {task_id} did not finish within timeout")


@pytest.mark.asyncio
async def test_plan_music_video_async_cancellation(client, sample_job):
    """Test cancelling an in-progress async planning task updates task status and unloads model."""
    import asyncio
    from app.services.video.video_orchestrator import video_orchestrator
    from app.services.video.types import VideoTaskStatusInfo
    task_id = f"plan_test_cancel_{uuid.uuid4().hex[:6]}"

    # Enqueue a mock planning task
    with video_orchestrator._lock:
        cancel_ev = asyncio.Event()
        video_orchestrator._plan_cancels[task_id] = cancel_ev
        video_orchestrator._tasks[task_id] = VideoTaskStatusInfo(
            id=task_id,
            job_id=str(sample_job.id),
            status="processing",
            step="AI Visual Director Conceptualizing",
            progress=50
        )

    # Cancel via API
    cancel_res = await client.post(f"/videos/tasks/{task_id}/cancel")
    assert cancel_res.status_code == 200
    assert cancel_res.json()["ok"] is True
    assert cancel_ev.is_set()

    # Verify task status is cancelled
    status = video_orchestrator.get_task(task_id)
    assert status is not None
    assert status["status"] == "cancelled"


@pytest.mark.asyncio
async def test_render_video_task_instant_asyncio_cancellation(client, sample_job):
    """Test cancelling an active background video render task cancels the running asyncio.Task object instantly."""
    import asyncio
    from app.services.video.video_orchestrator import video_orchestrator
    from app.services.video.types import VideoTaskStatusInfo

    task_id = f"render_test_cancel_{uuid.uuid4().hex[:6]}"

    async def _dummy_render_loop():
        try:
            while True:
                await asyncio.sleep(0.05)
        except asyncio.CancelledError:
            raise

    bg_task = asyncio.create_task(_dummy_render_loop())
    video_orchestrator.register_render_task(task_id, bg_task)

    with video_orchestrator._lock:
        cancel_ev = asyncio.Event()
        video_orchestrator._video_cancels[task_id] = cancel_ev
        video_orchestrator._tasks[task_id] = VideoTaskStatusInfo(
            id=task_id,
            job_id=str(sample_job.id),
            status="processing",
            step="Rendering Scene 1/8",
            progress=25
        )

    # Cancel via API
    cancel_res = await client.post(f"/videos/tasks/{task_id}/cancel")
    assert cancel_res.status_code == 200
    assert cancel_res.json()["ok"] is True
    assert cancel_ev.is_set()

    # The background task should be cancelled immediately
    await asyncio.sleep(0.05)
    assert bg_task.cancelled() or bg_task.done()

    # Verify task status is cancelled
    status = video_orchestrator.get_task(task_id)
    assert status is not None
    assert status["status"] == "cancelled"
    assert status["progress"] == 0


@pytest.mark.asyncio
async def test_wan_step_cancellation_callback():
    """Test DiffusersWanGenerator raises asyncio.CancelledError immediately if cancel_check triggers."""
    import asyncio
    from app.services.video.generators.diffusers_wan import DiffusersWanGenerator

    generator = DiffusersWanGenerator(model_size="1.3b")
    cancel_ev = asyncio.Event()
    cancel_ev.set()  # Pre-cancelled

    with pytest.raises(asyncio.CancelledError):
        await generator.generate_clip(
            prompt="A cinematic drone shot",
            duration=5.0,
            out_path="/tmp/test_nonexistent.mp4",
            cancel_event=cancel_ev,
            cancel_check=lambda: cancel_ev.is_set()
        )


@pytest.mark.asyncio
async def test_minimax_h3_step_cancellation():
    """Test MiniMaxH3Generator raises asyncio.CancelledError immediately when cancelled."""
    import asyncio
    from app.services.video.generators.minimax_h3 import MiniMaxH3Generator

    generator = MiniMaxH3Generator()
    cancel_ev = asyncio.Event()
    cancel_ev.set()

    with pytest.raises(asyncio.CancelledError):
        await generator.generate_clip(
            prompt="An epic cyberpunk city",
            duration=5.0,
            out_path="/tmp/test_minimax_cancel.mp4",
            cancel_event=cancel_ev,
            cancel_check=lambda: cancel_ev.is_set()
        )





