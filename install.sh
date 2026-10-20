#!/usr/bin/env bash
# One-time installer for whisperstt.
#
# Creates a private virtual environment, installs the package and its
# dependencies, links the `whisperstt` command into ~/.local/bin, and moves a
# previously downloaded model into its final location.

set -euo pipefail

APP="whisperstt"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/$APP"
CONFIG_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/$APP"
MODELS_DIR="$DATA_DIR/models"
VENV="$DATA_DIR/venv"
BIN_DIR="${XDG_BIN_HOME:-$HOME/.local/bin}"

info() { printf '\033[1m%s\033[0m\n' "$*"; }

# ---------------------------------------------------------------- Python ----
if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 is required but was not found." >&2
    exit 1
fi

if ! python3 -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 9) else 1)'; then
    echo "Python 3.9 or newer is required (found $(python3 -V 2>&1))." >&2
    exit 1
fi

# ---------------------------------------------------------------- FFmpeg ----
if ! command -v ffmpeg >/dev/null 2>&1; then
    if command -v pacman >/dev/null 2>&1; then
        info "Installing ffmpeg with pacman..."
        if ! sudo pacman -S --needed ffmpeg; then
            echo "Could not install ffmpeg. Install it manually and re-run." >&2
            exit 1
        fi
    else
        echo "ffmpeg is required. Install it with your package manager and re-run." >&2
        exit 1
    fi
fi

# ------------------------------------------------------- Virtual env -------
info "Creating virtual environment in $VENV"
mkdir -p "$DATA_DIR"
if [ ! -x "$VENV/bin/python" ]; then
    if command -v uv >/dev/null 2>&1; then
        uv venv --python python3 "$VENV"
    else
        python3 -m venv "$VENV"
    fi
fi

info "Installing whisperstt and dependencies"
if command -v uv >/dev/null 2>&1; then
    uv pip install --python "$VENV/bin/python" --upgrade "$SCRIPT_DIR"
else
    "$VENV/bin/python" -m pip install --upgrade pip >/dev/null
    "$VENV/bin/python" -m pip install --upgrade "$SCRIPT_DIR"
fi

# ---------------------------------------------------------- CUDA libs ------
if command -v nvidia-smi >/dev/null 2>&1; then
    if [ -z "$(find "$VENV" -name 'libcublas.so.12*' -print -quit 2>/dev/null)" ]; then
        info "Installing CUDA 12 runtime libraries for ctranslate2"
        if command -v uv >/dev/null 2>&1; then
            uv pip install --python "$VENV/bin/python" \
                nvidia-cublas-cu12 nvidia-cudnn-cu12
        else
            "$VENV/bin/python" -m pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
        fi
    fi
fi

# ------------------------------------------------------------- Command -----
mkdir -p "$BIN_DIR"
ln -sf "$VENV/bin/whisperstt" "$BIN_DIR/whisperstt"
info "Linked $BIN_DIR/whisperstt"

case ":$PATH:" in
    *":$BIN_DIR:"*) ;;
    *)
        echo
        echo "Note: $BIN_DIR is not on your PATH. Add this to your shell config:"
        echo "  export PATH=\"\$PATH:$BIN_DIR\""
        ;;
esac

# ------------------------------------------------------ Existing model -----
mkdir -p "$MODELS_DIR"
LEGACY="$SCRIPT_DIR/whisperModels/faster-whisper-large-v3"
if [ -d "$LEGACY" ] && [ ! -d "$MODELS_DIR/large-v3" ]; then
    info "Moving existing model to $MODELS_DIR/large-v3"
    mv "$LEGACY" "$MODELS_DIR/large-v3"
    rmdir "$SCRIPT_DIR/whisperModels" 2>/dev/null || true
fi

# Adopt a model that is already on disk so the user is not asked again.
mkdir -p "$CONFIG_DIR"
CONFIG_FILE="$CONFIG_DIR/config.json"
if [ ! -f "$CONFIG_FILE" ] && [ -d "$MODELS_DIR/large-v3" ]; then
    printf '{\n  "model": "large-v3"\n}\n' >"$CONFIG_FILE"
fi

# ---------------------------------------------------------- Model menu -----
if [ -t 0 ] && [ ! -f "$CONFIG_FILE" ]; then
    info "No model installed yet — choose one:"
    "$VENV/bin/whisperstt" --models
fi

echo
info "Done. Try: whisperstt /path/to/movie.mkv"
