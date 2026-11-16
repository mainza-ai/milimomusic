"""Capability matrix (decision D6).

MiniMax Music 3 generation is Apple-Silicon-MLX only. That fact used to live in a
README paragraph while the UI happily offered the engine on any machine and failed
at runtime. The matrix is now computed and served so the picker can disable
engines with a stated reason.
"""

from __future__ import annotations

import pytest

from app.core.capabilities import (
    APPLE_SILICON,
    CPU,
    CUDA,
    MATRIX,
    build_matrix,
    feature_supported,
    platform_class,
)


def test_matrix_covers_every_documented_capability():
    for feature in ("minimax_music3", "heartmula", "mulacover", "separation",
                    "transcription", "mastering", "image", "video", "cloud"):
        assert feature in MATRIX, f"{feature} is missing from the published matrix"


def test_every_row_has_all_three_columns():
    for feature, spec in MATRIX.items():
        for column in (APPLE_SILICON, CUDA, CPU):
            assert column in spec["columns"], f"{feature} has no {column} column"


def test_minimax_is_mlx_only():
    """D6: the flagship must not be offered where it cannot run."""
    row = MATRIX["minimax_music3"]["columns"]
    assert row[APPLE_SILICON] is True
    assert row[CUDA] is False
    assert row[CPU] is False


def test_every_unsupported_cell_has_a_reason():
    for feature, spec in MATRIX.items():
        for column, supported in spec["columns"].items():
            if supported:
                continue
            assert spec["reason"].get(column), (
                f"{feature} is unsupported on {column} but gives no reason"
            )


def test_build_matrix_reports_this_machine():
    payload = build_matrix()
    assert payload["platform"] in (APPLE_SILICON, CUDA, CPU)
    assert payload["platform"] == platform_class()
    assert isinstance(payload["ffmpeg"], bool)

    by_feature = {row["feature"]: row for row in payload["features"]}
    # A row is only "supported" when the platform allows it AND nothing is missing.
    for row in by_feature.values():
        if row["supported"]:
            assert row["reason"] is None
            assert row["matrix"].get(payload["platform"]) is True
        else:
            assert row["reason"], f"{row['feature']} is disabled without a reason"


def test_daw_paths_support_cpu():
    """D6: the DAW, separation, transcription and mastering run everywhere."""
    for feature in ("separation", "transcription", "mastering", "cloud"):
        assert MATRIX[feature]["columns"][CPU] is True


def test_feature_supported_matches_the_matrix():
    payload = build_matrix()
    by_feature = {row["feature"]: row for row in payload["features"]}
    for feature in MATRIX:
        # `feature_supported` is the cheap platform-only gate; the served row also
        # accounts for installed weights/keys, so it can only be stricter.
        if feature_supported(feature):
            assert by_feature[feature]["matrix"][payload["platform"]] is True


def test_unknown_feature_is_not_blocked():
    """An unknown name must not silently disable a feature."""
    assert feature_supported("something-not-in-the-matrix") is True
