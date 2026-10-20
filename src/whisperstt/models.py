"""Model catalogue: sizing, interactive selection and download."""

import os

from . import config
from .cuda import detect_gpu

MODEL_TABLE = (
    {
        "name": "tiny",
        "repo": "Systran/faster-whisper-tiny",
        "disk_mb": 75,
        "vram_mb": 800,
        "note": "fastest, lowest accuracy",
    },
    {
        "name": "base",
        "repo": "Systran/faster-whisper-base",
        "disk_mb": 141,
        "vram_mb": 1000,
        "note": "",
    },
    {
        "name": "small",
        "repo": "Systran/faster-whisper-small",
        "disk_mb": 464,
        "vram_mb": 1500,
        "note": "good speed/accuracy balance",
    },
    {
        "name": "medium",
        "repo": "Systran/faster-whisper-medium",
        "disk_mb": 1460,
        "vram_mb": 3000,
        "note": "",
    },
    {
        "name": "large-v3-turbo",
        "repo": "deepdml/faster-whisper-large-v3-turbo-ct2",
        "disk_mb": 1547,
        "vram_mb": 3000,
        "note": "near large-v3 quality, ~3x faster",
    },
    {
        "name": "large-v3",
        "repo": "Systran/faster-whisper-large-v3",
        "disk_mb": 2948,
        "vram_mb": 5000,
        "note": "best accuracy",
    },
)

# Fraction of total VRAM we are willing to fill before switching to int8.
VRAM_HEADROOM = 0.9


def get(name):
    for entry in MODEL_TABLE:
        if entry["name"] == name:
            return entry
    choices = ", ".join(entry["name"] for entry in MODEL_TABLE)
    raise SystemExit(f"Unknown model {name!r}. Available: {choices}")


def model_path(name):
    return config.models_dir() / name


def is_installed(name):
    return (model_path(name) / "model.bin").exists()


def format_size(megabytes):
    if megabytes >= 1024:
        return f"{megabytes / 1024:.1f} GB"
    return f"{megabytes} MB"


def default_model(gpu=None):
    """Pick the largest model that fits the detected GPU."""
    if gpu is None:
        return "small"
    budget = gpu[1] * VRAM_HEADROOM
    fitting = [entry for entry in MODEL_TABLE if entry["vram_mb"] <= budget]
    return fitting[-1]["name"] if fitting else "tiny"


def choose_model(preselected=None):
    gpu = detect_gpu()
    if gpu:
        print(f"Detected GPU: {gpu[0]} — {gpu[1] / 1024:.1f} GB VRAM")
    else:
        print("No NVIDIA GPU detected: models will run on CPU (much slower).")

    default = preselected or default_model(gpu)
    default_index = next(
        (i for i, entry in enumerate(MODEL_TABLE, 1) if entry["name"] == default), 1
    )

    print()
    print("  #  model             disk      VRAM     notes")
    for index, entry in enumerate(MODEL_TABLE, 1):
        too_big = bool(gpu) and entry["vram_mb"] > gpu[1] * VRAM_HEADROOM
        note = entry["note"]
        if too_big:
            note = (note + " " if note else "") + "(exceeds your GPU)"
        vram = "~" + format_size(entry["vram_mb"])
        print(
            f"  {index}  {entry['name']:<16}  "
            f"{format_size(entry['disk_mb']):>7}  "
            f"{vram:>8}  {note}"
        )

    while True:
        try:
            answer = input(f"\nChoose [{default_index}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            raise SystemExit("\nNo model selected.")
        if not answer:
            return MODEL_TABLE[default_index - 1]["name"]
        if answer.isdigit() and 1 <= int(answer) <= len(MODEL_TABLE):
            return MODEL_TABLE[int(answer) - 1]["name"]
        print("Enter one of the numbers above.")


def ensure_model(name, quiet=False):
    """Return the local directory for a model, downloading it if needed."""
    entry = get(name)
    target = model_path(name)
    if is_installed(name):
        return target

    target.mkdir(parents=True, exist_ok=True)
    if not quiet:
        print(
            f"Downloading {name} ({format_size(entry['disk_mb'])}) "
            f"from {entry['repo']}..."
        )

    # The Xet transfer backend has been observed to stall at 0 bytes on some
    # connections; the classic CDN download is slower but reliable.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    if quiet:
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    from huggingface_hub import snapshot_download

    try:
        snapshot_download(repo_id=entry["repo"], local_dir=str(target))
    except Exception as exc:
        raise SystemExit(f"Could not download {name}: {exc}")

    if not is_installed(name):
        raise SystemExit(
            f"Download of {name} is incomplete (model.bin missing) in {target}."
        )
    if not quiet:
        print(f"Model ready at {target}")
    return target


def compute_type_for(entry, device, gpu_vram_mb):
    """Choose a compute type that fits the available VRAM."""
    if device != "cuda":
        return "int8"
    if not gpu_vram_mb:
        return "float16"
    if entry["vram_mb"] <= gpu_vram_mb * VRAM_HEADROOM:
        return "float16"
    return "int8_float16"
