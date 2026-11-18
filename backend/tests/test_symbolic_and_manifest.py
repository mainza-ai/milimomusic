"""Symbolic-export honesty (plan 7.24) and the artifact manifest (plan 7.22)."""

from __future__ import annotations

import json
import uuid
import xml.etree.ElementTree as ET

import pytest
from sqlmodel import Session

from app.main import engine
from app.models import Job, JobStatus
from app.transcription.muscriptor_provider import MuScriptorProvider, _key_xml, _note_type


def _notes():
    return [
        {"pitch": 60, "start_time": 0.0, "end_time": 0.5, "duration": 0.5, "instrument": "Piano", "velocity": 90},
        {"pitch": 64, "start_time": 0.5, "end_time": 1.5, "duration": 1.0, "instrument": "Piano", "velocity": 80},
        {"pitch": 67, "start_time": 1.5, "end_time": 4.0, "duration": 2.5, "instrument": "Piano", "velocity": 70},
    ]


def test_score_does_not_assert_a_key_it_does_not_know():
    xml = MuScriptorProvider()._generate_musicxml(_notes(), bpm=120, title="T")
    root = ET.fromstring(xml)
    assert root.findall(".//key") == [], "an unknown key must not be rendered as C major"
    # Metre is only asserted when supplied; the default 4/4 is still declared,
    # so assert it matches what was asked for rather than a hardcoded literal.
    time = root.find(".//time")
    assert time is not None
    assert time.findtext("beats") == "4"


def test_score_carries_the_key_and_metre_it_is_given():
    xml = MuScriptorProvider()._generate_musicxml(
        _notes(), bpm=90, title="T", key="A minor", time_signature=(3, 4)
    )
    root = ET.fromstring(xml)
    assert root.find(".//key/fifths").text == "0"
    assert root.find(".//key/mode").text == "minor"
    assert root.find(".//time/beats").text == "3"
    assert root.find(".//time/beat-type").text == "4"


def test_encoding_date_is_not_a_hardcoded_literal():
    xml = MuScriptorProvider()._generate_musicxml(_notes(), bpm=120)
    date = ET.fromstring(xml).find(".//encoding-date").text
    assert date != "2026-08-21"
    assert len(date) == 10 and date[4] == "-"


def test_note_types_follow_the_duration():
    assert _note_type(4.0) == "whole"
    assert _note_type(1.0) == "quarter"
    assert _note_type(0.5) == "eighth"
    assert _note_type(0.2) == "32nd"


def test_key_xml_only_renders_known_keys():
    assert _key_xml(None) == ""
    assert _key_xml("H major") == ""
    assert "<fifths>-1</fifths>" in _key_xml("F major")


def test_every_note_carries_a_type_matching_its_duration():
    xml = MuScriptorProvider()._generate_musicxml(_notes(), bpm=120)
    root = ET.fromstring(xml)
    typed = [n for n in root.findall(".//note") if n.find("pitch") is not None]
    assert typed
    for note in typed:
        assert note.find("type") is not None
        assert note.find("type").text in {"whole", "half", "quarter", "eighth", "16th", "32nd"}


# ── manifest ─────────────────────────────────────────────────────────────

@pytest.fixture()
def manifest_job(tmp_path):
    job_uuid = uuid.uuid4()
    audio = tmp_path / "master.wav"
    audio.write_bytes(b"RIFF-fake")
    with Session(engine) as session:
        session.add(Job(
            id=job_uuid, status=JobStatus.COMPLETED, prompt="a song", title="Manifest test",
            audio_path=str(audio), seed=1234, temperature=0.9, cfg_scale=3.5, topk=50,
            model_provider="minimax_music3", quality="neural",
            provenance_json=json.dumps({"generation": {"source": "neural"}}),
        ))
        session.commit()
    yield job_uuid
    with Session(engine) as session:
        row = session.get(Job, job_uuid)
        if row:
            session.delete(row)
            session.commit()


def test_manifest_records_the_facts_needed_to_reproduce(manifest_job):
    from app.core.artifact_manifest import build_manifest

    with Session(engine) as session:
        job = session.get(Job, manifest_job)
        manifest = build_manifest(job)

    assert manifest["schema_version"]
    assert manifest["generation"]["seed"] == 1234
    assert manifest["generation"]["temperature"] == 0.9
    assert manifest["generation"]["model_prompt"] == "a song"
    assert manifest["provenance"]["generation"]["source"] == "neural"
    assert manifest["job"]["quality"] == "neural"
    assert manifest["application"]["python"]
    # The master is hashed, so a published artifact is verifiable.
    master = next(a for a in manifest["artifacts"] if a["kind"] == "master_audio")
    assert master["present"] is True
    assert len(master["sha256"]) == 64
    json.dumps(manifest)  # must be serialisable


def test_manifest_is_honest_about_missing_facts(manifest_job):
    from app.core.artifact_manifest import build_manifest

    with Session(engine) as session:
        job = session.get(Job, manifest_job)
        job.video_path = "/audio/videos/missing.mp4"
        manifest = build_manifest(job)

    video = next(a for a in manifest["artifacts"] if a["kind"] == "video")
    assert video["present"] is False, "a path that does not exist must not be reported as an artifact"
