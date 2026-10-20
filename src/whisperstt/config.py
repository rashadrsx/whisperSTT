import json
import os
from pathlib import Path

APP = "whisperstt"

DEFAULTS = {
    "model": None,
    "language": None,
    "output_dir": None,
    "device": "auto",
    "vad": True,
    "compute_type": "auto",
}


def _xdg(env_var, fallback):
    base = os.environ.get(env_var) or os.path.expanduser(fallback)
    return Path(base) / APP


def config_dir():
    return _xdg("XDG_CONFIG_HOME", "~/.config")


def data_dir():
    return _xdg("XDG_DATA_HOME", "~/.local/share")


def models_dir():
    return data_dir() / "models"


def venv_dir():
    return data_dir() / "venv"


def config_file():
    return config_dir() / "config.json"


def load():
    """Read the config file, falling back to defaults for missing keys."""
    settings = dict(DEFAULTS)
    path = config_file()
    if path.exists():
        try:
            settings.update(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            pass
    return settings


def save(settings):
    path = config_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    return path
