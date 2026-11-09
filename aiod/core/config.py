"""Configuration management for AIOD."""

import os
from pathlib import Path
from dataclasses import dataclass

BASE_DIR = Path(__file__).resolve().parent.parent.parent


def _load_env_file(env_path: Path):
    """Safely load .env file into os.environ if present."""
    if env_path.exists():
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, val = line.split("=", 1)
                        key = key.strip()
                        val = val.strip().strip("'\"")
                        if key and key not in os.environ:
                            os.environ[key] = val
        except Exception:
            pass


_load_env_file(BASE_DIR / ".env")


@dataclass(frozen=True)
class Settings:
    """Application runtime settings and defaults."""

    # Model configuration
    model_path: Path = Path(os.getenv("AIOD_MODEL_PATH", str(BASE_DIR / "yolo26n.pt")))
    default_confidence: float = float(os.getenv("AIOD_CONFIDENCE", "0.25"))
    default_iou: float = float(os.getenv("AIOD_IOU", "0.45"))
    device: str = os.getenv("AIOD_DEVICE", "")  # Empty string allows Ultralytics auto-selection

    # Gemini Vision API Key
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")

    # MySQL configuration (loaded strictly from environment variables)
    db_host: str = os.getenv("AIOD_DB_HOST", "localhost")
    db_port: int = int(os.getenv("AIOD_DB_PORT", "3306"))
    db_user: str = os.getenv("AIOD_DB_USER", "root")
    db_password: str = os.getenv("AIOD_DB_PASSWORD", "")
    db_name: str = os.getenv("AIOD_DB_NAME", "object_detection_db")

    # SQLite fallback configuration
    sqlite_db_path: Path = BASE_DIR / "aiod_database.db"
    enable_sqlite_fallback: bool = os.getenv("AIOD_SQLITE_FALLBACK", "true").lower() in ("1", "true", "yes")


settings = Settings()
