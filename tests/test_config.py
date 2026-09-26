from pathlib import Path

import pytest

from already_mentioned.config import DEFAULT_DATABASE_PATH, load_settings


def test_defaults_do_not_require_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("DATABASE_PATH", raising=False)
    settings = load_settings(env_file=tmp_path / "missing.env")
    assert settings.telegram_bot_token is None
    assert settings.database_path == DEFAULT_DATABASE_PATH
    with pytest.raises(ValueError, match="TELEGRAM_BOT_TOKEN"):
        settings.require_bot_token()


def test_environment_configuration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "  sample-token  ")
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "custom.db"))
    settings = load_settings(env_file=tmp_path / "missing.env")
    assert settings.require_bot_token() == "sample-token"
    assert settings.database_path == tmp_path / "custom.db"
