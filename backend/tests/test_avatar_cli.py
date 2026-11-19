"""External avatar engines: contract, availability and argv (plan 8.4).

These tests run without any weights: they pin the *contract*. If the argv shape or
availability logic drifts, the engine silently stops working for every user who
installed it, so this is worth locking down.
"""

from __future__ import annotations

import asyncio
import os

import pytest

from app.services.video.lip_sync import cli_provider as cli
from app.services.video.lip_sync import registry as reg


@pytest.fixture()
def fake_engine(tmp_path, monkeypatch):
    """A directory that looks like an installed EchoMimic checkout."""
    spec = cli.AVATAR_CLI_SPECS["echomimic_flash"]
    root = tmp_path / "echomimic"
    root.mkdir()
    (root / spec.script).write_text("# entry point")
    monkeypatch.setenv(spec.env_var, str(root))
    return spec, root


def test_an_engine_is_available_only_when_its_script_exists(fake_engine, monkeypatch):
    spec, root = fake_engine
    assert cli.spec_available(spec) is True
    assert cli.spec_reason(spec) is None

    os.remove(os.path.join(root, spec.script))
    assert cli.spec_available(spec) is False
    assert "not inside" in (cli.spec_reason(spec) or "")


def test_an_unset_env_var_explains_the_fix(monkeypatch):
    spec = cli.AVATAR_CLI_SPECS["wan_s2v"]
    monkeypatch.delenv(spec.env_var, raising=False)
    assert cli.spec_available(spec) is False
    assert spec.env_var in (cli.spec_reason(spec) or "")


def test_the_catalogue_lists_external_tiers_with_metadata(monkeypatch):
    monkeypatch.delenv("MILIMO_ECHOMIMIC_DIR", raising=False)
    engines = {e.id: e for e in reg.engine_catalogue()}
    assert "echomimic_flash" in engines
    assert "wan_s2v" in engines
    echomimic = engines["echomimic_flash"]
    assert echomimic.available is False
    assert "Apache-2.0" in (echomimic.licence or "")
    assert "MILIMO_ECHOMIMIC_DIR" in (echomimic.requires or "")
    # Wan S2V is the singing-trained tier and says so in its name.
    assert "singing" in engines["wan_s2v"].name.lower()


def test_resolution_routes_an_installed_engine_to_its_cli_provider(fake_engine):
    provider, reason = reg.resolve_local_provider("echomimic_flash")
    assert type(provider).__name__ == "CliAvatarProvider"
    assert provider.spec.id == "echomimic_flash"
    assert reason is None


def test_resolution_falls_back_with_a_reason_when_not_installed(monkeypatch):
    monkeypatch.delenv("MILIMO_ECHOMIMIC_DIR", raising=False)
    provider, reason = reg.resolve_local_provider("echomimic_flash")
    assert type(provider).__name__ == "SmoothVisemeFallbackProvider"
    assert reason and "unavailable" in reason.lower()


def test_argv_matches_the_engine_documented_contract(fake_engine):
    spec, root = fake_engine
    argv = cli.build_argv(
        spec,
        script_path=os.path.join(str(root), spec.script),
        image_path="/tmp/face.png",
        audio_path="/tmp/voice.wav",
        out_path="/tmp/out.mp4",
        prompt="a singer",
        duration=2.0,
        width=768, height=768,
    )
    assert argv[0] == "python"
    assert argv[1].endswith(spec.script)
    assert "--image_path" in argv and "/tmp/face.png" in argv
    assert "--audio_path" in argv and "/tmp/voice.wav" in argv
    assert "--save_path" in argv and "/tmp/out.mp4" in argv
    # 2 s at the engine's native rate, not the project rate.
    assert argv[argv.index("--video_length") + 1] == str(2 * spec.fps)
    assert argv[argv.index("--fps") + 1] == str(spec.fps)


def test_wan_s2v_argv_uses_its_task_and_constant_flags(monkeypatch, tmp_path):
    spec = cli.AVATAR_CLI_SPECS["wan_s2v"]
    root = tmp_path / "wan"
    root.mkdir()
    (root / spec.script).write_text("")
    monkeypatch.setenv(spec.env_var, str(root))

    argv = cli.build_argv(
        spec, script_path=str(root / spec.script),
        image_path="/tmp/f.png", audio_path="/tmp/a.wav", out_path="/tmp/o.mp4",
        prompt="singing", duration=5.0, width=1024, height=704,
    )
    assert argv[argv.index("--task") + 1] == "s2v-14B"
    assert argv[argv.index("--size") + 1] == "1024*704"
    assert argv[argv.index("--ckpt_dir") + 1] == str(root)


def test_a_missing_engine_reports_an_outcome_and_falls_back(fake_engine, monkeypatch):
    """A broken install must degrade loudly, not return a fake success."""
    spec, root = fake_engine
    provider = cli.CliAvatarProvider(spec)
    monkeypatch.delenv(spec.env_var, raising=False)

    ok = asyncio.run(provider.render_lip_sync(
        face_image_path="/tmp/face.png", vocal_audio_path="/tmp/a.wav",
        start_time=0.0, duration=1.0, out_path=str(root / "out.mp4"),
    ))
    assert ok is False
    assert provider.last_outcome["used_fallback"] is True
    assert provider.last_outcome["reason"]


# ── request-JSON launchers (InfiniteTalk, plan 8.4b) ──────────────────────

def test_json_template_preserves_the_engine_schema_and_fills_placeholders(tmp_path):
    template = {
        "cond_image": "{{image}}",
        "cond_audio": "{{audio}}",
        "prompt": "{{prompt}}",
        "n_sample_frames": "{{frames}}",
        "size": "infinitetalk-480",
        "mode": "streaming",
    }
    path = tmp_path / "template.json"
    path.write_text(__import__("json").dumps(template))

    payload = cli.render_json_request(
        str(path), image_path="/tmp/face.png", audio_path="/tmp/voice.wav",
        prompt="a singer on stage", duration=2.0, fps=25, out_path="/tmp/out.mp4",
    )
    data = __import__("json").loads(payload)
    assert data["cond_image"] == "/tmp/face.png"
    assert data["cond_audio"] == "/tmp/voice.wav"
    assert data["prompt"] == "a singer on stage"
    assert data["n_sample_frames"] == 50          # numeric, not a string
    # Engine-owned keys survive untouched.
    assert data["size"] == "infinitetalk-480"
    assert data["mode"] == "streaming"


def test_a_broken_template_returns_none_instead_of_crashing(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert cli.render_json_request(
        str(bad), image_path="a", audio_path="b", prompt="", duration=1.0,
        fps=25, out_path="c",
    ) is None


def test_infinitetalk_needs_both_the_script_and_a_template(tmp_path, monkeypatch):
    spec = cli.AVATAR_CLI_SPECS["infinitetalk"]
    root = tmp_path / "infinitetalk"
    root.mkdir()
    (root / spec.script).write_text("")
    monkeypatch.setenv(spec.env_var, str(root))
    monkeypatch.delenv(spec.json_template_env, raising=False)

    assert cli.spec_available(spec) is False
    assert spec.json_template_env in (cli.spec_reason(spec) or "")

    template = tmp_path / "req.json"
    template.write_text("{}")
    monkeypatch.setenv(spec.json_template_env, str(template))
    assert cli.spec_available(spec) is True


def test_infinitetalk_argv_uses_the_request_json(tmp_path, monkeypatch):
    spec = cli.AVATAR_CLI_SPECS["infinitetalk"]
    monkeypatch.setenv("MILIMO_AVATAR_REQUEST_JSON", "/tmp/req.json")
    argv = cli.build_argv(
        spec, script_path="/opt/infinitetalk/generate_infinitetalk.py",
        image_path="/tmp/f.png", audio_path="/tmp/a.wav", out_path="/tmp/o.mp4",
        prompt="x", duration=1.0, width=480, height=832,
    )
    assert argv[argv.index("--input_json") + 1] == "/tmp/req.json"
    assert argv[argv.index("--mode") + 1] == "streaming"
