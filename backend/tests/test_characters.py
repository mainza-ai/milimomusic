"""Character library (plan 8.3).

Consistency used to be per-render config; a character is the reusable half — a
description plus a content-addressed reference plate that any song can reference.
"""

from __future__ import annotations

import os
import uuid

import pytest
from PIL import Image
from sqlmodel import Session

from app.main import engine
from app.models import Character
from app.services import characters as chars


@pytest.fixture()
def plate(tmp_path):
    path = tmp_path / "portrait.png"
    Image.new("RGB", (64, 64), (120, 40, 200)).save(path)
    return str(path)


@pytest.fixture(autouse=True)
def clean_characters():
    with Session(engine) as session:
        for row in session.exec(__import__("sqlmodel").select(Character)).all():
            session.delete(row)
        session.commit()
    yield
    with Session(engine) as session:
        for row in session.exec(__import__("sqlmodel").select(Character)).all():
            session.delete(row)
        session.commit()


def test_create_character_copies_and_hashes_the_plate(plate):
    with Session(engine) as session:
        character = chars.create_character(
            session, name="Nova", description="tall, silver hair, leather jacket",
            plate_source=plate,
        )
        data = character.to_dict()

    assert data["name"] == "Nova"
    assert data["description"].startswith("tall")
    assert data["has_plate"] is True
    assert len(data["plate_sha256"]) == 64
    assert os.path.isfile(data["plate_path"])
    assert data["plate_path"].endswith(os.path.splitext(plate)[1])


def test_the_same_portrait_is_not_duplicated_on_disk(plate):
    with Session(engine) as session:
        first = chars.create_character(session, name="A", plate_source=plate)
        second = chars.create_character(session, name="B", plate_source=plate)
        assert first.plate_path == second.plate_path
        assert first.plate_sha256 == second.plate_sha256


def test_a_missing_image_is_refused_not_stored_broken(tmp_path):
    with Session(engine) as session:
        with pytest.raises(ValueError):
            chars.create_character(session, name="Ghost", plate_source=str(tmp_path / "nope.png"))


def test_an_unsupported_extension_is_refused(tmp_path):
    bad = tmp_path / "portrait.txt"
    bad.write_text("not an image")
    with Session(engine) as session:
        with pytest.raises(ValueError):
            chars.create_character(session, name="Bad", plate_source=str(bad))


def test_a_description_without_a_plate_is_allowed():
    with Session(engine) as session:
        character = chars.create_character(session, name="Text only", description="green eyes")
        assert character.to_dict()["has_plate"] is False
        resolved = chars.resolve_for_render(session, character.id)
        assert resolved["description"] == "green eyes"
        assert resolved["plate_path"] is None


def test_update_and_delete_clean_up_the_plate(plate):
    with Session(engine) as session:
        character = chars.create_character(session, name="Old", plate_source=plate)
        cid = character.id
        plate_path = character.plate_path

        updated = chars.update_character(session, cid, name="New", description="changed")
        assert updated.name == "New"
        assert updated.description == "changed"

        assert chars.delete_character(session, cid) is True
        assert chars.get_character(session, cid) is None
        assert not os.path.isfile(plate_path), "an unshared plate must be removed"


def test_a_shared_plate_survives_deleting_one_character(plate):
    with Session(engine) as session:
        first = chars.create_character(session, name="Shared one", plate_source=plate)
        second = chars.create_character(session, name="Shared two", plate_source=plate)
        shared_path = first.plate_path

        assert chars.delete_character(session, first.id) is True
        assert os.path.isfile(shared_path), "the other character still needs this plate"
        assert chars.get_character(session, second.id) is not None


def test_unknown_character_ids_resolve_to_nothing():
    with Session(engine) as session:
        assert chars.resolve_for_render(session, str(uuid.uuid4())) == {}
        assert chars.resolve_for_render(session, "not-a-uuid") == {}
        assert chars.delete_character(session, "not-a-uuid") is False


# ── endpoints ─────────────────────────────────────────────────────────────

def test_character_endpoints_round_trip(client, plate):
    created = client.post("/characters", json={
        "name": "API Nova", "description": "short, red coat", "plate_source": plate,
    })
    assert created.status_code == 201, created.text
    body = created.json()["character"]
    cid = body["id"]

    listed = client.get("/characters")
    assert listed.status_code == 200
    assert any(c["id"] == cid for c in listed.json()["characters"])

    patched = client.patch(f"/characters/{cid}", json={"description": "updated look"})
    assert patched.status_code == 200
    assert patched.json()["character"]["description"] == "updated look"

    deleted = client.delete(f"/characters/{cid}")
    assert deleted.status_code == 200
    assert client.delete(f"/characters/{cid}").status_code == 404


def test_character_endpoint_rejects_a_missing_image(client):
    res = client.post("/characters", json={"name": "Bad", "plate_source": "/nope/missing.png"})
    assert res.status_code == 400
    assert res.json()["detail"]["error"]["code"] == "invalid_character"
