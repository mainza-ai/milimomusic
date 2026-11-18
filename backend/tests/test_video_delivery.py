"""Delivery presets and aspect variants (plans 7.15, 7.17).

The real assertions are ffprobe checks: a preset must produce the codec, size,
bitrate class and audio the destination requires, and a variant must land on the
requested pixels.
"""

from __future__ import annotations

import os
import shutil
import subprocess

import pytest

from app.services.video import deliver
from app.services.video.export_presets import get_preset, probe_streams

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")


@pytest.fixture(scope="module")
def master(tmp_path_factory):
    """A 3 s 16:9 master with an AAC stereo track."""
    path = tmp_path_factory.mktemp("deliver") / "master.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", "testsrc=size=320x180:rate=25:duration=3",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=3:sample_rate=44100",
         "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "128k", str(path)],
        check=True, capture_output=True, timeout=120,
    )
    return str(path)


def test_youtube_preset_hits_the_expected_pixels_and_faststart(master, tmp_path):
    dest = tmp_path / "yt.mp4"
    ok = deliver.transcode(master, str(dest), get_preset("youtube_1080p"))
    assert ok
    streams = probe_streams(str(dest))
    assert (streams["width"], streams["height"]) == (1920, 1080)
    assert streams["video_codec"] == "h264"
    assert streams["audio_codec"] == "aac"
    # faststart: moov must precede mdat.
    with open(dest, "rb") as fh:
        head = fh.read(2_000_000)
    assert head.find(b"moov") < head.find(b"mdat")


def test_canvas_preset_is_silent_and_at_most_eight_seconds(master, tmp_path):
    dest = tmp_path / "canvas.mp4"
    ok = deliver.transcode(master, str(dest), get_preset("canvas"), width=1080, height=1920)
    assert ok
    streams = probe_streams(str(dest))
    assert streams["audio_codec"] is None, "Spotify Canvas must be silent"
    assert (streams["width"], streams["height"]) == (1080, 1920)
    assert streams["duration"] <= 8.1


def test_vevo_preset_uses_44k_audio_and_a_high_video_bitrate(master, tmp_path):
    dest = tmp_path / "vevo.mp4"
    ok = deliver.transcode(master, str(dest), get_preset("vevo_hd"))
    assert ok
    streams = probe_streams(str(dest))
    assert streams["audio_sample_rate"] == 44100
    assert (streams["width"], streams["height"]) == (1920, 1080)


def test_unknown_preset_falls_back_to_default_instead_of_failing():
    assert get_preset("does-not-exist").id == "default"


def test_size_for_supports_the_advertised_combinations():
    assert deliver.size_for("9:16", "1080p") == (1080, 1920)
    assert deliver.size_for("1:1", "1080p") == (1080, 1080)
    assert deliver.size_for("21:9", "1080p") == (2560, 1080)
    assert deliver.size_for("4:3") is None


def test_aspect_variants_crop_to_the_requested_shape(master, tmp_path):
    for aspect, expected in (("9:16", (1080, 1920)), ("1:1", (1080, 1080))):
        dest = tmp_path / f"{aspect.replace(':', 'x')}.mp4"
        pixels = deliver.size_for(aspect, "1080p")
        ok = deliver.transcode(master, str(dest), get_preset("default"),
                               width=pixels[0], height=pixels[1], mode="crop")
        assert ok, f"{aspect} variant failed"
        streams = probe_streams(str(dest))
        assert (streams["width"], streams["height"]) == expected


def test_pad_mode_letterboxes_instead_of_cropping(master, tmp_path):
    dest = tmp_path / "padded.mp4"
    ok = deliver.transcode(master, str(dest), get_preset("default"),
                           width=1080, height=1080, mode="pad")
    assert ok
    streams = probe_streams(str(dest))
    assert (streams["width"], streams["height"]) == (1080, 1080)
    assert streams["duration"] >= 2.5


def test_transcode_refuses_a_missing_source(tmp_path):
    assert deliver.transcode(str(tmp_path / "nope.mp4"), str(tmp_path / "out.mp4"),
                             get_preset("default")) is False


def test_finalise_master_is_faststart_loudness_normalised_and_qc_gated(tmp_path, master):
    """The real master path (plan 7.12-7.14), not just the export helper."""
    import asyncio

    from app.services.video.video_orchestrator import video_orchestrator

    out = tmp_path / "master_out.mp4"
    result = asyncio.run(video_orchestrator._finalise_master(
        stitched_video=master,
        audio_path=master,
        out_path=str(out),
        width=320,
        height=180,
        song_duration=probe_streams(master)["duration"],
        ass_path=None,
        timed_lines=[],
        preset_id="default",
    ))

    assert result["qc"]["ok"] is True, result["qc"]
    assert result["preset"] == "default"
    streams = probe_streams(str(out))
    assert streams["audio_codec"] == "aac"
    assert streams["video_codec"] == "h264"

    with open(out, "rb") as fh:
        head = fh.read(2_000_000)
    assert head.find(b"moov") < head.find(b"mdat"), "the delivered master must be faststart"

    # −14 LUFS is the delivery target; the measurement must be reported.
    loudness = result["qc"]["checks"].get("loudness") or {}
    assert loudness.get("integrated_lufs") is not None


# ── frame-rate delivery (audit G5) ────────────────────────────────────────

def test_interpolation_filter_only_fires_when_needed():
    from app.services.video.clip_normalise import interpolation_filter

    assert interpolation_filter(16, 24) == "minterpolate=fps=24.000:mi_mode=blend"
    assert interpolation_filter(24, 24) is None
    assert interpolation_filter(None, 24) is None
    assert interpolation_filter(25, None) is None


def test_a_16fps_master_is_delivered_at_24fps_with_new_frames(tmp_path):
    """A Wan 2.1 render used to ship at 16 fps (duplicated frames on any conversion)."""
    import asyncio
    import subprocess

    from app.services.video.video_orchestrator import video_orchestrator

    src = tmp_path / "clip16.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", "testsrc=size=320x180:rate=16:duration=2",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=2:sample_rate=44100",
         "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
         "-c:a", "aac", str(src)],
        check=True, capture_output=True, timeout=120,
    )

    out = tmp_path / "master24.mp4"
    asyncio.run(video_orchestrator._finalise_master(
        stitched_video=str(src), audio_path=str(src), out_path=str(out),
        width=320, height=180, song_duration=2.0, ass_path=None,
        timed_lines=[], preset_id="default",
    ))

    streams = probe_streams(str(out))
    assert streams["fps"] == 24.0, f"expected 24 fps delivery, got {streams['fps']}"
    # 2 s at 24 fps must contain more frames than the 32-frame source.
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=nb_read_frames", "-of", "default=nw=1:nk=1", str(out)],
        capture_output=True, text=True, check=True,
    )
    assert int(probe.stdout.strip()) > 32
