"""Incremental single-scene retake (plan 7.11).

The old behaviour re-diffused every scene to fix one shot, because the per-scene
renders were deleted after assembly. This test proves the retake reuses the other
scenes, re-encodes the new one, produces a NEW take, and leaves the old master
alone.
"""

from __future__ import annotations

import asyncio
import os
import uuid

import pytest
from sqlmodel import Session

from app.main import engine
from app.models import Job, JobStatus
from app.services.video import takes as takes_service
from app.services.video.video_orchestrator import video_orchestrator as orch


@pytest.fixture()
def job():
    job_uuid = uuid.uuid4()
    with Session(engine) as session:
        session.add(Job(id=job_uuid, status=JobStatus.COMPLETED, prompt="t",
                        title="Retake test", audio_path="/audio/x.wav", lyrics="[Verse]\nhello"))
        session.commit()
    yield job_uuid
    with Session(engine) as session:
        row = session.get(Job, job_uuid)
        if row:
            session.delete(row)
            session.commit()


@pytest.fixture()
def prepared_take(job, tmp_path, monkeypatch):
    """A completed take whose workspace holds 3 clips and a 3-scene plan."""
    from app.services.video import takes

    monkeypatch.setattr(takes, "clip_workspace", lambda rid: str(tmp_path / f"take-{rid}"))

    with Session(engine) as session:
        take = takes_service.create_take(session, job_id=str(job), task_id="orig",
                                         kind="advanced", config={"visual_style": "neon-cyberpunk",
                                                                  "model_name": "wan_14b",
                                                                  "provider": "local",
                                                                  "width": 320, "height": 180})
        takes_service.finish_take(session, take, status="completed",
                                  video_path="/audio/videos/orig.mp4", file_path=str(tmp_path / "orig.mp4"))
        take_id = str(take.id)

    clips_dir = tmp_path / f"take-{take_id}" / "clips"
    clips_dir.mkdir(parents=True)
    clip_paths = []
    for i in range(3):
        p = clips_dir / f"clip_{i:03d}.mp4"
        p.write_bytes(f"clip-{i}".encode())
        clip_paths.append(str(p))

    scenes = [
        {"clip_index": i + 1, "start_time": float(i * 3), "end_time": float((i + 1) * 3),
         "duration": 3.0, "prompt": f"scene {i + 1}", "is_vocal": False,
         "scene_type": "CINEMATIC_BROLL", "time_str": f"{i * 3}s", "camera": "", "lighting": ""}
        for i in range(3)
    ]
    takes_service.save_clip_plan(take_id, {"scenes": scenes, "clips": clip_paths,
                                           "config": {"visual_style": "neon-cyberpunk",
                                                      "model_name": "wan_14b"}}, clip_paths)
    return {"take_id": take_id, "clips": clip_paths, "scenes": scenes, "tmp": tmp_path}


def _stub_pipeline(monkeypatch, prepared, calls):
    monkeypatch.setattr(orch, "resolve_audio_path", lambda p: str(prepared["tmp"] / "song.wav"))
    monkeypatch.setattr(orch, "resolve_vocals_stem", lambda job: None)
    monkeypatch.setattr(orch, "resolve_face_image", lambda job, custom=None: None)
    (prepared["tmp"] / "song.wav").write_bytes(b"RIFF")

    class FakeGen:
        name = "fake-diffusion"

        async def generate_clip(self, prompt, duration, out_path, **kwargs):
            calls.append({"prompt": prompt, "out_path": out_path})
            with open(out_path, "wb") as fh:
                fh.write(b"new-clip")
            return True

    monkeypatch.setattr(
        "app.services.video.generator_registry.VideoGeneratorRegistry.resolve",
        classmethod(lambda cls, model, provider: FakeGen()),
    )

    def fake_normalise(src, dest, **kwargs):
        calls.append({"normalise": (src, dest, kwargs.get("duration_s"))})
        with open(dest, "wb") as fh:
            fh.write(b"normalised")
        return True

    monkeypatch.setattr("app.services.video.clip_normalise.normalise_clip", fake_normalise)
    monkeypatch.setattr("app.services.video.clip_normalise.probe_duration", lambda p: 9.0)

    async def fake_stitch(rendered_clips, transition_style, stitched_video, task_id):
        calls.append({"stitched_from": list(rendered_clips)})
        with open(stitched_video, "wb") as fh:
            fh.write(b"stitched")
        return True

    monkeypatch.setattr(orch, "_stitch_video_segments_with_transitions", fake_stitch)

    async def fake_finalise(**kwargs):
        calls.append({"finalise": kwargs["out_path"]})
        with open(kwargs["out_path"], "wb") as fh:
            fh.write(b"master")
        return {"qc": {"ok": True, "warnings": []}, "captions": {}, "preset": "default"}

    monkeypatch.setattr(orch, "_finalise_master", fake_finalise)
    return orch


def test_retake_rerenders_one_scene_and_reuses_the_rest(job, prepared_take, monkeypatch):
    calls = []
    orch = _stub_pipeline(monkeypatch, prepared_take, calls)

    url = asyncio.run(orch.render_scene_retake(
        job=type("J", (), {"id": job, "audio_path": "/audio/x.wav", "lyrics": "",
                            "duration_ms": 9000})(),
        task_id="retake-task",
        render_id=prepared_take["take_id"],
        clip_index=2,
        config={"prompt": "a close-up of hands on keys", "burn_lyrics": False,
                "width": 320, "height": 180},
    ))

    assert url.endswith(".mp4")
    # Exactly one diffusion pass, and it targeted scene 2.
    assert len([c for c in calls if "prompt" in c]) == 1
    assert calls[0]["prompt"] == "a close-up of hands on keys"

    # Assembly used the two untouched clips plus the new one.
    stitched_from = next(c["stitched_from"] for c in calls if "stitched_from" in c)
    assert stitched_from[0] == prepared_take["clips"][0]
    assert stitched_from[2] == prepared_take["clips"][2]
    assert stitched_from[1] != prepared_take["clips"][1], "scene 2 must be the new render"

    # A new take exists, is selected, and the old one is intact.
    with Session(engine) as session:
        takes = takes_service.list_takes(session, str(job))
        kinds = {t["kind"] for t in takes}
        assert "retake" in kinds and "advanced" in kinds
        selected = [t for t in takes if t["is_selected"]]
        assert len(selected) == 1 and selected[0]["kind"] == "retake"


def test_retake_without_persisted_clips_says_so(job, monkeypatch):
    with pytest.raises(ValueError) as exc:
        asyncio.run(orch.render_scene_retake(
            job=type("J", (), {"id": job, "audio_path": "/audio/x.wav", "lyrics": "",
                                "duration_ms": 9000})(),
            task_id="retake-task-2",
            render_id=str(uuid.uuid4()),
            clip_index=1,
            config={},
        ))
    assert "render the full video" in str(exc.value).lower()
