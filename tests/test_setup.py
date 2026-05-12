import json
from unittest.mock import patch

from scripts import setup as setup_mod


def test_status_for_returns_ready_when_everything_present():
    with patch.object(setup_mod, "_which", side_effect=lambda x: f"/bin/{x}"):
        with patch.object(setup_mod, "_read_env", return_value={"GROQ_API_KEY": "k"}):
            s = setup_mod.status_for()
    assert s["status"] == "ready"
    assert s["missing_binaries"] == []
    assert s["has_api_key"] is True


def test_status_for_flags_missing_binaries():
    def fake_which(x):
        return f"/bin/{x}" if x == "ffprobe" else None
    with patch.object(setup_mod, "_which", side_effect=fake_which):
        with patch.object(setup_mod, "_read_env", return_value={"GROQ_API_KEY": "k"}):
            s = setup_mod.status_for()
    assert s["status"] == "needs_install"
    assert "ffmpeg" in s["missing_binaries"]
    assert "yt-dlp" in s["missing_binaries"]


def test_status_for_flags_missing_key():
    def fake_which(x):
        return None if x == "whisper-cli" else f"/bin/{x}"
    with patch.object(setup_mod, "_which", side_effect=fake_which):
        with patch.object(setup_mod, "_read_env", return_value={}):
            with patch.object(setup_mod, "_resolve_local_model", return_value=None):
                s = setup_mod.status_for()
    assert s["status"] == "needs_key"
    assert s["has_api_key"] is False
    assert s["has_local_whisper"] is False


def test_status_for_ready_with_local_whisper_no_key():
    with patch.object(setup_mod, "_which", side_effect=lambda x: f"/bin/{x}"):
        with patch.object(setup_mod, "_read_env", return_value={}):
            with patch.object(
                setup_mod, "_resolve_local_model", return_value="/fake/model.bin"
            ):
                s = setup_mod.status_for()
    assert s["status"] == "ready"
    assert s["has_api_key"] is False
    assert s["has_local_whisper"] is True
    assert s["whisper_backend"] == "local"


def test_resolve_local_model_env_var_wins(tmp_path):
    """An explicit WHISPER_CPP_MODEL beats every default path."""
    target = tmp_path / "custom.bin"
    target.write_bytes(b"x")
    got = setup_mod._resolve_local_model({"WHISPER_CPP_MODEL": str(target)})
    assert got == target


def test_resolve_local_model_env_var_missing_returns_none(tmp_path):
    """An explicit WHISPER_CPP_MODEL pointing at a missing file returns None.

    We do NOT silently fall back to the defaults — if the user named a path,
    honor exactly that path. Falling back would mask a user typo.
    """
    target = tmp_path / "does-not-exist.bin"
    got = setup_mod._resolve_local_model({"WHISPER_CPP_MODEL": str(target)})
    assert got is None


def test_resolve_local_model_falls_through_extra_paths_when_default_missing(monkeypatch, tmp_path):
    """When DEFAULT_LOCAL_MODEL is absent and an EXTRA path exists, use the EXTRA path.

    This is the sibling-installer integration case: creativity-maxxing's media
    module drops the model at ~/.whisper/ggml-base.en.bin, not at
    ~/.config/watch/models/ggml-base.en.bin.
    """
    missing_default = tmp_path / "default-missing" / "model.bin"
    present_extra = tmp_path / "extra-present" / "model.bin"
    present_extra.parent.mkdir(parents=True)
    present_extra.write_bytes(b"x")
    monkeypatch.setattr(setup_mod, "DEFAULT_LOCAL_MODEL", missing_default)
    monkeypatch.setattr(setup_mod, "EXTRA_LOCAL_MODEL_PATHS", (present_extra,))
    got = setup_mod._resolve_local_model({})
    assert got == present_extra


def test_resolve_local_model_default_wins_over_extra(monkeypatch, tmp_path):
    """If BOTH default and extra exist, the watch default wins."""
    present_default = tmp_path / "default" / "model.bin"
    present_extra = tmp_path / "extra" / "model.bin"
    present_default.parent.mkdir(parents=True)
    present_extra.parent.mkdir(parents=True)
    present_default.write_bytes(b"x")
    present_extra.write_bytes(b"x")
    monkeypatch.setattr(setup_mod, "DEFAULT_LOCAL_MODEL", present_default)
    monkeypatch.setattr(setup_mod, "EXTRA_LOCAL_MODEL_PATHS", (present_extra,))
    got = setup_mod._resolve_local_model({})
    assert got == present_default


def test_resolve_local_model_none_when_no_paths_exist(monkeypatch, tmp_path):
    monkeypatch.setattr(setup_mod, "DEFAULT_LOCAL_MODEL", tmp_path / "no-default.bin")
    monkeypatch.setattr(
        setup_mod, "EXTRA_LOCAL_MODEL_PATHS", (tmp_path / "no-extra.bin",)
    )
    assert setup_mod._resolve_local_model({}) is None


def test_status_for_combines_when_both_missing():
    with patch.object(setup_mod, "_which", return_value=None):
        with patch.object(setup_mod, "_read_env", return_value={}):
            s = setup_mod.status_for()
    assert s["status"] == "needs_install_and_key"


def test_check_exit_code_maps_status_to_table():
    assert setup_mod.exit_code_for("ready") == 0
    assert setup_mod.exit_code_for("needs_install") == 2
    assert setup_mod.exit_code_for("needs_key") == 3
    assert setup_mod.exit_code_for("needs_install_and_key") == 4


def test_json_output_shape(capsys):
    with patch.object(setup_mod, "_which", side_effect=lambda x: f"/bin/{x}"):
        with patch.object(setup_mod, "_read_env", return_value={"GROQ_API_KEY": "k"}):
            rc = setup_mod.main(["--check", "--json"])
    captured = capsys.readouterr().out
    payload = json.loads(captured)
    assert payload["status"] == "ready"
    assert "platform" in payload
    assert rc == 0


def test_check_silent_on_success(capsys):
    with patch.object(setup_mod, "_which", side_effect=lambda x: f"/bin/{x}"):
        with patch.object(setup_mod, "_read_env", return_value={"GROQ_API_KEY": "k"}):
            rc = setup_mod.main(["--check"])
    out = capsys.readouterr().out
    assert out == "" and rc == 0
