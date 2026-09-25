"""
Unit and integration tests for Phase 3: Non-Destructive Multi-Track Timeline & Immersive Gallery Media Bridge.
"""

import tempfile
from pathlib import Path

import pytest

from app.services.gallery.media_bridge import MediaBridge, RoutedMediaPayload
from app.services.timeline.editor_projects import (
    ClipTransform,
    EditorProject,
    EditorProjectManager,
    TimelineClip,
    TimelineTrack,
)


def test_editor_project_schema_roundtrip():
    project = EditorProject(
        project_id="test_proj_001",
        title="Synthwave Music Video",
        duration=120.0,
        aspect_ratio="16:9",
        resolution=(1920, 1080),
        fps=24,
        tracks=[
            TimelineTrack(
                track_id="track_v1",
                track_type="video",
                name="Main Video Track",
                volume=1.0,
                clips=[
                    TimelineClip(
                        clip_id="clip_01",
                        asset_path="assets/take1.mp4",
                        start_time=0.0,
                        duration=5.0,
                        source_in=0.0,
                        transform=ClipTransform(scale=1.0),
                    )
                ],
            ),
            TimelineTrack(
                track_id="track_a1",
                track_type="audio",
                name="Vocals Stem",
                volume=0.9,
                clips=[
                    TimelineClip(
                        clip_id="clip_a01",
                        asset_path="audio/vocals.wav",
                        start_time=0.0,
                        duration=120.0,
                    )
                ],
            ),
        ],
    )

    data = project.to_dict()
    assert data["project_id"] == "test_proj_001"
    assert len(data["tracks"]) == 2

    reconstructed = EditorProject.from_dict(data)
    assert reconstructed.project_id == project.project_id
    assert reconstructed.tracks[0].clips[0].clip_id == "clip_01"


def test_compile_editor_render_command():
    project = EditorProject(
        project_id="test_render_01",
        title="Test Render",
        duration=10.0,
        tracks=[
            TimelineTrack(
                track_id="v1",
                track_type="video",
                name="Video 1",
                clips=[
                    TimelineClip(
                        clip_id="c1",
                        asset_path="clip1.mp4",
                        start_time=0.0,
                        duration=5.0,
                    ),
                    TimelineClip(
                        clip_id="c2",
                        asset_path="clip2.mp4",
                        start_time=5.0,
                        duration=5.0,
                    ),
                ],
            ),
            TimelineTrack(
                track_id="a1",
                track_type="audio",
                name="Audio 1",
                clips=[
                    TimelineClip(
                        clip_id="ac1",
                        asset_path="audio.wav",
                        start_time=0.0,
                        duration=10.0,
                    )
                ],
            ),
        ],
    )

    cmd = EditorProjectManager.compile_editor_render_command(project, "output.mp4")
    assert "ffmpeg" in cmd[0]
    assert "-filter_complex" in cmd
    # Verify filter complex connects inputs
    filter_idx = cmd.index("-filter_complex") + 1
    filter_graph = cmd[filter_idx]
    assert "[outv]" in filter_graph
    assert "[outa]" in filter_graph


def test_ai_take_roundtrip_replacement():
    project = EditorProject(
        project_id="test_retake",
        title="Retake Test",
        duration=10.0,
        tracks=[
            TimelineTrack(
                track_id="v1",
                track_type="video",
                name="Video",
                clips=[
                    TimelineClip(
                        clip_id="target_clip",
                        asset_path="original_take.mp4",
                        start_time=2.0,
                        duration=4.0,
                        take_version=1,
                    )
                ],
            )
        ],
    )

    # Apply retake
    success = EditorProjectManager.apply_retake_to_clip(
        project, "target_clip", "new_ai_take_v2.mp4"
    )
    assert success
    clip = project.tracks[0].clips[0]
    assert clip.asset_path == "new_ai_take_v2.mp4"
    assert clip.ai_take_parent_id == "original_take.mp4"
    assert clip.take_version == 2
    # Ensure timing was untouched
    assert clip.start_time == 2.0
    assert clip.duration == 4.0


def test_gallery_media_bridge_routing():
    with tempfile.NamedTemporaryFile(suffix=".png") as tmp_img:
        routed = MediaBridge.route_to_input(
            media_path=tmp_img.name,
            target_slot="references",
            target_job_or_session_id="job_abc123",
        )
        assert isinstance(routed, RoutedMediaPayload)
        assert routed.media_type == "image"
        assert routed.target_slot == "references"
        assert routed.target_job_or_session_id == "job_abc123"

    split_manifest = MediaBridge.create_before_after_split_manifest(
        source_url="http://localhost:8000/source.mp4",
        generated_url="http://localhost:8000/generated.mp4",
        split_ratio=0.5,
    )
    assert split_manifest["type"] == "split_comparison"
    assert split_manifest["initial_split_ratio"] == 0.5
