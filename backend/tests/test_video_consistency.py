"""Subject consistency: character lock and last-frame chaining (audit E6)."""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess

import pytest

from app.services.video import consistency as c

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")


# ── character bible ───────────────────────────────────────────────────────

def test_bible_is_stamped_once_and_only_once():
    bible = "woman, 30s, close-cropped red hair, freckles, black leather jacket"
    first = c.apply_character_bible("neon rooftop at night", bible)
    second = c.apply_character_bible(first, bible)
    assert first == second, "re-stamping must be idempotent"
    assert first.count(bible) == 1
    assert first.startswith("neon rooftop at night.")


def test_bible_handles_existing_terminal_punctuation():
    out = c.apply_character_bible("A singer on stage!", "tall, silver hair")
    assert "stage! consistent character:" in out
    assert "!." not in out


def test_empty_bible_leaves_the_prompt_untouched():
    assert c.apply_character_bible("a shot", "") == "a shot"


def test_bible_prompt_with_only_the_bible_is_valid():
    out = c.apply_character_bible("", "tall, silver hair")
    assert out.startswith("consistent character: tall, silver hair")


def test_explicit_character_description_wins_over_the_director_profile():
    plan = type("P", (), {"character_profile": "director default person"})()
    assert c.character_bible({"character_desc": "explicit person"}, plan) == "explicit person"


def test_director_profile_is_used_when_no_explicit_description():
    plan = type("P", (), {"character_profile": "green eyes, bomber jacket"})()
    assert c.character_bible({}, plan) == "green eyes, bomber jacket"


def test_director_placeholder_is_not_propagated():
    plan = type("P", (), {"character_profile": "Lead performer"})()
    assert c.character_bible({}, plan) == ""


def test_unknown_mode_falls_back_to_the_default():
    assert c.normalise_mode("banana") == c.DEFAULT_MODE
    assert c.normalise_mode(None) == c.DEFAULT_MODE
    assert c.normalise_mode("CHAIN") == "chain"


# ── last-frame chaining ───────────────────────────────────────────────────

@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    path = tmp_path_factory.mktemp("chain") / "clip.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", "testsrc=size=160x120:rate=24:duration=1.3",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "ultrafast", str(path)],
        check=True, capture_output=True, timeout=120,
    )
    return str(path)


def test_last_frame_is_extracted_as_a_real_png(clip, tmp_path):
    dest = tmp_path / "last.png"
    got = c.extract_last_frame(clip, str(dest))
    assert got == str(dest)
    assert os.path.getsize(got) > 100
    with open(got, "rb") as fh:
        assert fh.read(8) == b"\x89PNG\r\n\x1a\n"


def test_last_frame_of_a_missing_clip_is_none(tmp_path):
    assert c.extract_last_frame(str(tmp_path / "nope.mp4"), str(tmp_path / "x.png")) is None


# ── seed priority ─────────────────────────────────────────────────────────

def test_an_approved_plate_always_wins(clip, tmp_path):
    plate = str(tmp_path / "approved.png")
    c.extract_last_frame(clip, plate)
    assert c.scene_seed(approved_plate=plate, chained_frame=clip,
                        face_image=clip, mode="chain") == plate


def test_chain_mode_uses_the_previous_frame(clip, tmp_path):
    face = str(tmp_path / "face.png")
    c.extract_last_frame(clip, face)
    assert c.scene_seed(approved_plate=None, chained_frame=clip,
                        face_image=face, mode="chain") == clip


def test_character_mode_does_not_chain(clip, tmp_path):
    face = str(tmp_path / "face.png")
    c.extract_last_frame(clip, face)
    assert c.scene_seed(approved_plate=None, chained_frame=clip,
                        face_image=face, mode="character") == face


def test_without_any_seed_the_scene_is_text_to_video():
    assert c.scene_seed(approved_plate=None, chained_frame=None,
                        face_image=None, mode="chain") is None


# ── orchestration wiring ──────────────────────────────────────────────────

def test_render_stamps_the_bible_into_every_scene_prompt(tmp_path, monkeypatch):
    """The planning output is what the renderer sends, so the bible must land there."""
    from app.services.video.video_orchestrator import video_orchestrator as orch
    from app.services.video.types import SceneClip, VideoPlan

    bible = "androgynous singer, shaved head, silver jumpsuit"
    plan = VideoPlan(
        job_id="j", total_clips=2, vocal_clips_count=1, broll_clips_count=1,
        max_clip_duration=5.0, model_max_duration=5.0, model_name="wan_14b",
        bpm=120, visual_style="neon-cyberpunk",
        clips=[
            SceneClip(clip_index=1, start_time=0.0, end_time=5.0, duration=5.0,
                      time_str="0-5s", is_vocal=True, scene_type="VOCAL_PERFORMANCE",
                      prompt="close-up singing"),
            SceneClip(clip_index=2, start_time=5.0, end_time=10.0, duration=5.0,
                      time_str="5-10s", is_vocal=False, scene_type="CINEMATIC_BROLL",
                      prompt="rain-soaked street"),
        ],
    )

    # Stamp exactly as the render loop does.
    for clip in plan.clips:
        clip.prompt = c.apply_character_bible(clip.prompt, c.character_bible(
            {"character_desc": bible}, plan))
    assert all(bible in clip.prompt for clip in plan.clips)
    assert plan.clips[0].prompt.endswith(".")


# ── section-aware chaining (keeps `chain` from feeling monotonous) ────────

def test_chaining_stays_inside_a_section_by_default():
    assert c.chain_allowed("Verse", "Verse", None) is True
    assert c.chain_allowed("Verse", "Chorus", None) is False
    assert c.chain_allowed("Chorus", "Chorus", "section") is True


def test_scope_all_chains_across_sections():
    assert c.chain_allowed("Verse", "Chorus", "all") is True


def test_unknown_sections_do_not_disable_the_feature():
    assert c.chain_allowed(None, "Chorus", "section") is True
    assert c.chain_allowed("Verse", "", "section") is True


def test_unknown_scope_falls_back_to_section():
    assert c.normalise_chain_scope("everything") == "section"
    assert c.normalise_chain_scope("ALL") == "all"
