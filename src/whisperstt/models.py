"""Model catalogue: sizing, interactive selection and download."""

import os
import sys

from . import config, theme
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
    raise SystemExit(
        theme.paint(
            f"Unknown model {name!r}. Available: {choices}",
            "accent",
            "bold",
            stream=sys.stderr,
        )
    )


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
    print(theme.banner("choose a model"))

    if gpu:
        print(
            f"  {theme.mark('bullet')} {theme.paint(gpu[0], 'primary', 'bold')} "
            f"{theme.paint(f'· {gpu[1] / 1024:.1f} GB VRAM', 'muted')}"
        )
    else:
        print(
            f"  {theme.mark('bullet')} "
            f"{theme.paint('No NVIDIA GPU', 'primary', 'bold')} "
            f"{theme.paint('· models will run on CPU (much slower)', 'muted')}"
        )

    default = preselected or default_model(gpu)
    default_index = next(
        (i for i, entry in enumerate(MODEL_TABLE, 1) if entry["name"] == default), 1
    )

    print()
    header = (
        f"  {'':1} {'#':>2}  {'model':<16}  {'disk':>7}  {'VRAM':>8}  notes"
    )
    print(theme.paint(header, "muted"))
    print("  " + theme.paint("─" * 60, "rule"))

    for index, entry in enumerate(MODEL_TABLE, 1):
        too_big = bool(gpu) and entry["vram_mb"] > gpu[1] * VRAM_HEADROOM
        selected = entry["name"] == default
        marker = "▸" if selected else " "
        number = f"{index:>2}"
        name = f"{entry['name']:<16}"
        disk = f"{format_size(entry['disk_mb']):>7}"
        vram = f"{'~' + format_size(entry['vram_mb']):>8}"

        notes = []
        if entry["note"]:
            notes.append(theme.paint(entry["note"], "muted"))
        if too_big:
            notes.append(theme.paint("exceeds your GPU", "accent"))
        if is_installed(entry["name"]):
            notes.append(
                theme.paint(
                    "installed", "bg:light_greige", "on_surface", "bold"
                )
            )

        line = (
            f"  {theme.paint(marker, 'accent' if selected else 'muted')} "
            f"{theme.paint(number, 'accent' if selected else 'muted')}  "
            f"{theme.paint(name, 'primary', 'bold' if selected else 'dim')}  "
            f"{theme.paint(disk, 'muted')}  {theme.paint(vram, 'muted')}"
        )
        if notes:
            line += "  " + "  ".join(notes)
        print(line)

    while True:
        prompt = (
            theme.paint("  Choose model", "primary", "bold")
            + " "
            + theme.paint(
                str(default_index), "bg:terracotta", "on_accent", "bold"
            )
            + theme.paint(": ", "primary")
        )
        try:
            answer = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            raise SystemExit(
                theme.paint(
                    "No model selected.",
                    "accent",
                    "bold",
                    stream=sys.stderr,
                )
            )
        if not answer:
            return MODEL_TABLE[default_index - 1]["name"]
        if answer.isdigit() and 1 <= int(answer) <= len(MODEL_TABLE):
            return MODEL_TABLE[int(answer) - 1]["name"]
        print(
            f"  {theme.mark('warn')} "
            f"{theme.paint('Enter one of the numbers above.', 'muted')}"
        )


def ensure_model(name, quiet=False):
    """Return the local directory for a model, downloading it if needed."""
    entry = get(name)
    target = model_path(name)
    if is_installed(name):
        return target

    target.mkdir(parents=True, exist_ok=True)
    if not quiet:
        size = format_size(entry["disk_mb"])
        print(
            f"  {theme.mark('step')} {theme.paint('Downloading', 'primary')} "
            f"{theme.paint(name, 'accent', 'bold')} "
            f"{theme.paint(f'({size})', 'muted')} "
            f"{theme.paint('from', 'muted')} {theme.paint(entry['repo'], 'muted')}"
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
        raise SystemExit(
            theme.paint(
                f"Could not download {name}: {exc}",
                "accent",
                "bold",
                stream=sys.stderr,
            )
        )

    if not is_installed(name):
        raise SystemExit(
            theme.paint(
                f"Download of {name} is incomplete (model.bin missing) in {target}.",
                "accent",
                "bold",
                stream=sys.stderr,
            )
        )
    if not quiet:
        print(
            f"  {theme.mark('ok')} {theme.paint('Model ready at', 'primary')} "
            f"{theme.paint(str(target), 'muted')}"
        )
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
