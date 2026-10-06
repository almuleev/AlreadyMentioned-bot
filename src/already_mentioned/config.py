"""Runtime configuration, loaded only when explicitly requested."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_DATABASE_PATH = Path("data/already_mentioned.db")
# Provisional FRIDA calibration; explicit per-chat settings remain authoritative.
DEFAULT_SIMILARITY_THRESHOLD = 0.458


@dataclass(frozen=True, slots=True)
class Settings:
    telegram_bot_token: str | None
    database_path: Path

    def require_bot_token(self) -> str:
        """Return the token or explain how to configure bot startup."""
        if not self.telegram_bot_token:
            raise ValueError(
                "TELEGRAM_BOT_TOKEN is missing. Copy .env.example to .env "
                "and set TELEGRAM_BOT_TOKEN before starting the bot."
            )
        return self.telegram_bot_token


def load_settings(*, env_file: str | Path = ".env") -> Settings:
    """Read environment variables without requiring a token for offline tests."""
    load_dotenv(dotenv_path=env_file, override=False)
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip() or None
    database_path = Path(os.getenv("DATABASE_PATH") or DEFAULT_DATABASE_PATH)
    return Settings(telegram_bot_token=token, database_path=database_path)
