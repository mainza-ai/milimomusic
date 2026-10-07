"""Scene background pre-generation: batching, prompt caching, people-free plates.

Covers `ImageService.pregenerate_scene_backgrounds` and the video orchestrator's
`_prebuild_scene_plates` / `_scene_plate_is_current` contract — the pass that
diffuses every storyboard plate in ONE image-model load ahead of the clip loop.
"""
import asyncio
import os
import types

import pytest
from PIL import Image

from app.services import image_service as isvc
from app.services.image_service import image_service

PROMPTS = ["sunset over the savanna", "neon city alley rain", "close up of drummer hands"]


@pytest.fixture
def counts(tmp_path, monkeypatch):
    """Stub the diffusion core so tests count loads/renders instead of diffusing."""
    monkeypatch.setattr(isvc, "SCENE_STILLS_DIR", str(tmp_path / "scene_stills"))
    os.makedirs(str(tmp_path / "scene_stills"), exist_ok=True)
    counters = {"render": 0, "unload": 0, "resolve": 0}

    def fake_render(full_prompt, width, height, chosen_model_id, local_path, repo_id,
                    is_installed, dest_path, steps=None, guidance=None, log_label="image"):
        counters["render"] += 1
        assert "no people" in full_prompt, f"people-free suffix missing: {full_prompt}"
        Image.new("RGB", (int(width), int(height)), (10, 20, 30)).save(dest_path)
        return {"engine": "fake", "diffusion_error": None}

    def fake_unload():
        counters["unload"] += 1
        return True

    def fake_resolve(model_id=None):
        counters["resolve"] += 1
        return "fake_model", {"name": "Fake"}, None, "fake/repo", True

    monkeypatch.setattr(image_service, "_render_diffusion_image", fake_render)
    monkeypatch.setattr(image_service, "unload_models", fake_unload)
    monkeypatch.setattr(image_service, "_resolve_image_model", fake_resolve)
    return counters


def _pregen(**kw):
    kw.setdefault("style", "documentary")
    kw.setdefault("width", 1280)
    kw.setdefault("height", 720)
    return image_service.pregenerate_scene_backgrounds(**kw)


def test_batch_renders_every_scene_with_one_load_and_unload(counts):
    res = _pregen(job_id="jobA", prompts=PROMPTS)

    assert res["ok"] and res["generated"] == 3 and res["failed"] == 0
    assert counts == {"render": 3, "unload": 1, "resolve": 1}
    assert sorted(res["stills"]) == [0, 1, 2]
    for path in res["stills"].values():
        assert os.path.isfile(path) and os.path.isfile(path + ".prompt")


def test_unchanged_scenes_are_reused_without_diffusion(counts):
    _pregen(job_id="jobA", prompts=PROMPTS)
    counts["render"] = 0

    res = _pregen(job_id="jobA", prompts=PROMPTS)

    assert res["reused"] == 3 and res["generated"] == 0
    assert counts["render"] == 0


def test_rewritten_director_prompt_rediffuses_only_that_scene(counts):
    _pregen(job_id="jobA", prompts=PROMPTS)
    counts["render"] = 0

    res = _pregen(job_id="jobA", prompts=[PROMPTS[0], "REWRITTEN SCENE", PROMPTS[2]])

    assert (res["generated"], res["reused"], counts["render"]) == (1, 2, 1)


def test_visual_style_change_invalidates_every_plate(counts):
    _pregen(job_id="jobA", prompts=PROMPTS)
    counts["render"] = 0

    res = _pregen(job_id="jobA", prompts=PROMPTS, style="anime")

    assert res["generated"] == 3 and counts["render"] == 3


def test_wrong_frame_size_is_not_reused(counts):
    _pregen(job_id="jobA", prompts=PROMPTS[:1])
    counts["render"] = 0

    res = _pregen(job_id="jobA", prompts=PROMPTS[:1], width=720, height=720)

    assert res["generated"] == 1 and counts["render"] == 1


def test_cancel_check_stops_within_one_scene(counts):
    ticks = {"n": 0}

    def cancel_after_two():
        ticks["n"] += 1
        return ticks["n"] > 2

    res = _pregen(job_id="jobB", prompts=PROMPTS, cancel_check=cancel_after_two)

    assert res["cancelled"] and res["generated"] == 2 and res["ok"] is False


def test_index_start_keeps_one_cache_file_per_scene(counts):
    res = _pregen(job_id="jobC", prompts=PROMPTS[:1], index_start=4)

    assert res["stills"][0].endswith("scene_pregen_jobC_004.png")


def test_auto_unload_false_keeps_pipeline_hot(counts):
    _pregen(job_id="jobD", prompts=PROMPTS[:1], auto_unload=False)

    assert counts["unload"] == 0


def test_progress_events_arrive_in_scene_order(counts):
    events = []

    _pregen(job_id="jobE", prompts=PROMPTS[:2], progress_cb=events.append)

    assert [e["scene_index"] for e in events if e["status"] == "rendering"] == [0, 1]
    assert all(e["phase"] == "scene_prebuild" for e in events)


# --------------------------------------------------------------------------
# Orchestrator: batched pre-build + plate validation
# --------------------------------------------------------------------------

Clips = [
    types.SimpleNamespace(clip_index=1, prompt="dunes at dawn", is_vocal=False),
    types.SimpleNamespace(clip_index=2, prompt="singer on stage", is_vocal=True),
    types.SimpleNamespace(clip_index=3, prompt="lone car on ridge", is_vocal=False),
]


@pytest.fixture
def plates(tmp_path, monkeypatch):
    """Stub image batch rendering + keyframe dir; returns (batches, keyframes_dir)."""
    import importlib
    vo = importlib.import_module("app.services.video.video_orchestrator")

    keyframes = tmp_path / "keyframes"
    keyframes.mkdir()
    monkeypatch.setattr(vo, "KEYFRAMES_DIR", str(keyframes))
    stills = tmp_path / "stills"
    stills.mkdir()
    batches = []

    def fake_batch(job_id, prompts, style, width, height, progress_cb=None, cancel_check=None, **kw):
        batches.append({"job_id": job_id, "prompts": list(prompts), "style": style,
                        "width": width, "height": height})
        stills_map = {}
        for i, _p in enumerate(prompts):
            path = stills / f"batch_{job_id}_{i}.png"
            Image.new("RGB", (int(width), int(height)), (3, 4, 5)).save(str(path))
            stills_map[i] = str(path)
            if progress_cb:
                progress_cb({"phase": "scene_prebuild", "scene_index": i, "status": "rendered"})
        return {"ok": True, "cancelled": False, "total": len(prompts), "generated": len(prompts),
                "reused": 0, "failed": 0, "stills": stills_map, "engine": "fake",
                "diffusion_error": None}

    monkeypatch.setattr(isvc, "image_service", types.SimpleNamespace(
        pregenerate_scene_backgrounds=fake_batch,
        prompt_fingerprint=isvc.ImageService.prompt_fingerprint,
        stamp_scene_plate=isvc.ImageService.stamp_scene_plate,
    ))
    return batches, str(keyframes)


def _prebuild(clips, style="film noir", width=1280, height=720):
    import importlib
    vo = importlib.import_module("app.services.video.video_orchestrator")
    video_orchestrator = vo.video_orchestrator

    steps = []

    def fake_update(task_id, **kw):
        steps.append(kw)

    original = video_orchestrator.update_task
    video_orchestrator.update_task = fake_update
    # Fresh accelerator lock: each asyncio.run() gets its own loop, and an
    # asyncio.Lock binds to the loop it was first awaited on.
    from app.core.hardware_lock import GlobalHardwareCoordinator
    previous_lock = GlobalHardwareCoordinator._lock
    GlobalHardwareCoordinator._lock = asyncio.Lock()
    try:
        staged = asyncio.run(video_orchestrator._prebuild_scene_plates(
            types.SimpleNamespace(id="jobO"), clips, task_id="t1", style=style,
            width=width, height=height, cancel_event=asyncio.Event(),
        ))
    finally:
        video_orchestrator.update_task = original
        GlobalHardwareCoordinator._lock = previous_lock
    return staged, steps


def test_prebuild_batches_every_cinematic_scene_in_one_load(plates):
    batches, keyframes = plates

    staged, steps = _prebuild(Clips)

    assert [b["prompts"] for b in batches] == [["dunes at dawn", "lone car on ridge"]]
    assert sorted(staged) == [1, 3]  # vocal scene never seeds Wan i2v
    for path in staged.values():
        assert os.path.isfile(path) and os.path.isfile(path + ".prompt")
    labels = [s.get("step", "") for s in steps]
    assert any(lbl.startswith("Pre-building scene backgrounds... 1/2") for lbl in labels), labels
    assert "Pre-building scene backgrounds... 2/2 (plate ready)" in labels


def test_prebuild_skips_entirely_when_every_plate_is_current(plates):
    batches, keyframes = plates
    _prebuild(Clips)
    batches.clear()

    staged, _steps = _prebuild(Clips)

    assert batches == [] and staged == {}


def test_plate_without_sidecar_may_not_seed_wan(plates):
    _batches, keyframes = plates
    _prebuild(Clips[:1])
    kf = os.path.join(keyframes, "keyframe_jobO_001.png")

    from app.services.video.video_orchestrator import VideoOrchestrator
    assert VideoOrchestrator._scene_plate_is_current(kf, 1280, 720, "dunes at dawn", "film noir")
    os.remove(kf + ".prompt")
    assert not VideoOrchestrator._scene_plate_is_current(kf, 1280, 720, "dunes at dawn", "film noir")


def test_plate_fingerprint_covers_prompt_and_style(plates):
    _batches, keyframes = plates
    _prebuild(Clips[:1])
    kf = os.path.join(keyframes, "keyframe_jobO_001.png")

    from app.services.video.video_orchestrator import VideoOrchestrator
    assert VideoOrchestrator._scene_plate_is_current(kf, 1280, 720, "rewritten scene", "film noir") is False
    assert VideoOrchestrator._scene_plate_is_current(kf, 1280, 720, "dunes at dawn", "anime") is False


def _plate_with_size(keyframes, name, size, prompt="dunes at dawn", style="film noir"):
    path = os.path.join(keyframes, name)
    Image.new("RGB", size).save(path)
    with open(path + ".prompt", "w", encoding="utf-8") as fh:
        fh.write(isvc.ImageService.prompt_fingerprint(prompt, style))
    return path


def test_square_and_undersized_plates_are_rejected(plates):
    from app.services.video.video_orchestrator import VideoOrchestrator
    _batches, keyframes = plates

    square = _plate_with_size(keyframes, "square.png", (1024, 1024))
    assert VideoOrchestrator._scene_plate_is_current(square, 1280, 720, "dunes at dawn", "film noir") is False

    smaller = _plate_with_size(keyframes, "small.png", (768, 432))
    assert VideoOrchestrator._scene_plate_is_current(smaller, 1280, 720, "dunes at dawn", "film noir") is True

    tiny = _plate_with_size(keyframes, "tiny.png", (256, 144))
    assert VideoOrchestrator._scene_plate_is_current(tiny, 1280, 720, "dunes at dawn", "film noir") is False


def test_user_approved_retake_survives_the_prebuild_pass(plates):
    """A hand-picked retake is final, even though its prompt differs from the plan."""
    from app.services.video.video_orchestrator import VideoOrchestrator
    batches, keyframes = plates

    retake = os.path.join(keyframes, "keyframe_jobO_001.png")
    Image.new("RGB", (1280, 720), (200, 30, 90)).save(retake)
    assert isvc.ImageService.stamp_scene_plate(
        retake, "Explosive slow-motion rain falling upward", "film noir", approved=True
    ) is True

    # Storyboard prompt for this scene is completely different — the plate survives.
    assert VideoOrchestrator._scene_plate_is_current(retake, 1280, 720, "dunes at dawn", "film noir")

    staged, _steps = _prebuild([Clips[0]])

    assert batches == [] and staged == {}  # nothing to render, nothing staged over
    assert Image.open(retake).getpixel((0, 0)) == (200, 30, 90)  # retake pixels untouched

    # ...but the approved stamp does not excuse the wrong frame shape.
    tall = os.path.join(keyframes, "keyframe_jobO_003.png")
    Image.new("RGB", (720, 1280)).save(tall)
    isvc.ImageService.stamp_scene_plate(tall, "whatever", "film noir", approved=True)
    assert VideoOrchestrator._scene_plate_is_current(tall, 1280, 720, "lone car on ridge", "film noir") is False


def test_plain_mismatched_stamp_is_re_diffused(plates):
    """A batch-rendered plate with a different prompt is NOT treated as approved."""
    from app.services.video.video_orchestrator import VideoOrchestrator
    batches, keyframes = plates

    plate = os.path.join(keyframes, "keyframe_jobO_001.png")
    Image.new("RGB", (1280, 720)).save(plate)
    isvc.ImageService.stamp_scene_plate(plate, "an older director prompt", "film noir")
    assert VideoOrchestrator._scene_plate_is_current(plate, 1280, 720, "dunes at dawn", "film noir") is False

    _staged, _steps = _prebuild([Clips[0]])

    assert [b["prompts"] for b in batches] == [["dunes at dawn"]]  # re-diffused
