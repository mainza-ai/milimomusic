"""Licence classification and audit (audit finding G8).

The trigger for this module is real: the catalogue records a licence per model and
the product's effective image default turned out to be a **non-commercial** FLUX.2
variant while the Apache-2.0 4B sibling sat unused. A deployer must be able to see
that without reading 300 lines of catalogue.
"""

from __future__ import annotations

import pytest

from app.core import licensing as lic


@pytest.mark.parametrize("licence", [
    "Apache-2.0", "apache 2.0", "MIT", "BSD-3-Clause", "CC0",
])
def test_permissive_licences_are_recognised(licence):
    assert lic.classify(licence) == lic.PERMISSIVE
    assert lic.is_commercially_usable(licence) is True


@pytest.mark.parametrize("licence", [
    "FLUX.2 Non-Commercial License",
    "FLUX.1-dev Non-Commercial License",
    "CC BY-NC 4.0",
    "Stability AI Non-Commercial Research",
    "research only",
])
def test_non_commercial_licences_are_recognised(licence):
    assert lic.classify(licence) == lic.NON_COMMERCIAL
    assert lic.is_commercially_usable(licence) is False


@pytest.mark.parametrize("licence", [
    "MiniMax H3 Community License", "Stability AI Community License",
    "OpenRAIL++", "Custom",
])
def test_community_licences_are_conditional_not_permissive(licence):
    assert lic.classify(licence) == lic.CONDITIONAL


def test_unknown_is_never_assumed_permissive():
    assert lic.classify("Totally Bespoke Terms") == lic.UNKNOWN
    assert lic.classify(None) == lic.UNKNOWN
    assert lic.is_commercially_usable("Totally Bespoke Terms") is False


def test_a_non_commercial_default_is_a_blocker():
    catalogue = [
        {"id": "flux_9b", "category": "image", "license": "FLUX.2 Non-Commercial License"},
        {"id": "flux_4b", "category": "image", "license": "Apache-2.0"},
    ]
    report = lic.audit_catalogue(catalogue, defaults={"image": "flux_9b"})
    assert report["ok"] is False
    assert [b["id"] for b in report["blockers"]] == ["flux_9b"]


def test_a_permissive_default_is_not_a_blocker():
    catalogue = [
        {"id": "flux_4b", "category": "image", "license": "Apache-2.0"},
        {"id": "flux_9b", "category": "image", "license": "FLUX.2 Non-Commercial License"},
    ]
    report = lic.audit_catalogue(catalogue, defaults={"image": "flux_4b"})
    assert report["ok"] is True
    # The non-commercial model is still classified and visible.
    assert any(m["id"] == "flux_9b" and m["class"] == lic.NON_COMMERCIAL for m in report["models"])


def test_catalogue_is_default_flags_count_as_reachable():
    catalogue = [
        {"id": "wan_1_3b", "category": "video", "license": "Apache-2.0", "is_default": True},
    ]
    report = lic.audit_catalogue(catalogue)
    assert report["defaults"]["video"] == "wan_1_3b"
    assert report["ok"] is True


def test_live_catalogue_audit_returns_a_shape_the_endpoint_can_serve():
    report = lic.audit_catalogue(lic.collect_catalogue(), defaults=lic.resolve_effective_defaults())
    assert "models" in report and "blockers" in report and "counts" in report
    for model in report["models"]:
        assert model["class"] in {
            lic.PERMISSIVE, lic.CONDITIONAL, lic.NON_COMMERCIAL, lic.UNKNOWN,
        }


def test_licence_endpoint_serves_the_audit(client):
    res = client.get("/system/licences")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert isinstance(body["models"], list)
    assert "ok" in body and "blockers" in body


def test_a_deliberate_non_commercial_deployment_can_acknowledge(monkeypatch):
    """Milimo is non-commercial by design; the gate is a notice, not a wall."""
    catalogue = [{"id": "flux_9b", "category": "image", "license": "FLUX.2 Non-Commercial License"}]
    monkeypatch.delenv("MILIMO_NONCOMMERCIAL_OK", raising=False)
    assert lic.audit_catalogue(catalogue, defaults={"image": "flux_9b"})["ok"] is False

    monkeypatch.setenv("MILIMO_NONCOMMERCIAL_OK", "1")
    report = lic.audit_catalogue(catalogue, defaults={"image": "flux_9b"})
    assert report["ok"] is True
    assert report["acknowledged"] is True
    # The finding itself is still reported — acknowledgement is not amnesia.
    assert [b["id"] for b in report["blockers"]] == ["flux_9b"]
