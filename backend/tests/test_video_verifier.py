"""Plate verification and reshoot selection (plan 8.1).

Dependency-free so it works on a machine with no model downloads and is
deterministic in CI: difference hash + palette + Laplacian sharpness.
"""

from __future__ import annotations

import os

import pytest
from PIL import Image, ImageDraw, ImageFilter

from app.services.video import verifier


def _image(path, *, pattern="stripes", blur=0.0, brightness=128):
    img = Image.new("L", (128, 128), brightness)
    draw = ImageDraw.Draw(img)
    if pattern == "stripes":
        for x in range(0, 128, 8):
            draw.rectangle([x, 0, x + 4, 127], fill=230)
    elif pattern == "half":
        draw.rectangle([0, 0, 63, 127], fill=240)
    elif pattern == "dots":
        for y in range(8, 128, 16):
            for x in range(8, 128, 16):
                draw.ellipse([x, y, x + 5, y + 5], fill=250)
    if blur:
        img = img.filter(ImageFilter.GaussianBlur(blur))
    img.save(path)
    return str(path)


def test_identical_images_score_identically(tmp_path):
    a = _image(tmp_path / "a.png")
    b = _image(tmp_path / "b.png")
    score = verifier.score_candidate(a, b)
    assert score.identity > 0.99
    assert score.total > 0.9


def test_a_different_composition_scores_lower(tmp_path):
    ref = _image(tmp_path / "ref.png", pattern="stripes")
    other = _image(tmp_path / "other.png", pattern="half")
    same = verifier.score_candidate(_image(tmp_path / "same.png", pattern="stripes"), ref)
    diff = verifier.score_candidate(other, ref)
    assert same.identity > diff.identity
    assert same.total > diff.total


def test_blur_reduces_sharpness(tmp_path):
    sharp = verifier.score_candidate(_image(tmp_path / "s.png"), None)
    blurry = verifier.score_candidate(_image(tmp_path / "b.png", blur=4.0), None)
    assert sharp.sharpness > blurry.sharpness


def test_select_best_prefers_the_reference_like_candidate(tmp_path):
    ref = _image(tmp_path / "ref.png", pattern="stripes")
    good = _image(tmp_path / "good.png", pattern="stripes")
    bad = _image(tmp_path / "bad.png", pattern="half")
    best, scores = verifier.select_best([bad, good], ref)
    assert best == good
    assert len(scores) == 2


def test_select_best_with_no_candidates_is_none():
    best, scores = verifier.select_best([], None)
    assert best is None and scores == []


def test_reshoot_only_when_a_reference_was_used(tmp_path):
    ref = _image(tmp_path / "ref.png", pattern="stripes")
    bad = verifier.score_candidate(_image(tmp_path / "bad.png", pattern="half"), ref)
    assert verifier.needs_reshoot(bad, 0.55) is True

    # No reference: identity/palette are neutral, so "low score" is not a
    # consistency failure and must not trigger extra diffusion.
    unreferenced = verifier.score_candidate(_image(tmp_path / "x.png"), None)
    assert verifier.needs_reshoot(unreferenced, 0.9) is False


def test_a_strong_match_is_not_reshot(tmp_path):
    ref = _image(tmp_path / "ref.png", pattern="stripes")
    same = verifier.score_candidate(_image(tmp_path / "same.png", pattern="stripes"), ref)
    assert verifier.needs_reshoot(same, 0.55) is False


def test_unreadable_paths_do_not_raise(tmp_path):
    missing = str(tmp_path / "nope.png")
    assert verifier.difference_hash(missing) is None
    score = verifier.score_candidate(missing, missing)
    assert score.total == 0.0
