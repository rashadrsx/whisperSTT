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

# ---------------------------------------------------------------- Palette ---
# Warm parchment theme: the same five colours as the Python TUI.
if [ -z "${NO_COLOR:-}" ] &&
    { [ -n "${FORCE_COLOR:-}" ] ||
        { [ -t 1 ] && [ "${TERM:-dumb}" != "dumb" ]; }; }; then
    C_RESET=$'\033[0m'
    C_BOLD=$'\033[1m'
    C_ACCENT=$'\033[38;2;217;119;87m'
    C_SLATE=$'\033[38;2;177;173;161m'
    C_GREIGE=$'\033[38;2;232;230;220m'
    THEME_BG="${WHISPERSTT_BACKGROUND:-}"
    if [ -z "$THEME_BG" ]; then
        case "${COLORFGBG:-}" in
            *\;[7-9]|*\;1[0-5]) THEME_BG=light ;;
            *) THEME_BG=dark ;;
        esac
    fi
    case "$THEME_BG" in
        light|cream)
            C_TEXT=$'\033[38;2;20;20;19m'
            C_RULE="$C_GREIGE"
            ;;
        *)
            C_TEXT=$'\033[38;2;250;249;245m'
            C_RULE="$C_SLATE"
            ;;
    esac
else
    C_RESET="" C_BOLD="" C_ACCENT="" C_SLATE="" C_GREIGE=""
    C_TEXT="" C_RULE=""
fi
C_HEAD="${C_BOLD}${C_TEXT}"
C_ERR="${C_BOLD}${C_ACCENT}"

banner() {
    printf '\n  %s▌%s %swhisperstt%s\n' \
        "$C_ACCENT" "$C_RESET" "$C_HEAD" "$C_RESET"
    printf '    %s%s%s\n' "$C_SLATE" "${1:-install}" "$C_RESET"
    printf '  %s%s%s\n\n' "$C_RULE" \
        "────────────────────────────────────────────────────" "$C_RESET"
}
step() { printf '  %s›%s %s%s%s\n' "$C_ACCENT" "$C_RESET" "$C_TEXT" "$*" "$C_RESET"; }
ok() { printf '  %s✓%s %s%s%s\n' "$C_ACCENT" "$C_RESET" "$C_TEXT" "$*" "$C_RESET"; }
note() { printf '    %s%s%s\n' "$C_SLATE" "$*" "$C_RESET"; }
err() { printf '  %s✗%s %s%s%s\n' "$C_ACCENT" "$C_RESET" "$C_ERR" "$*" "$C_RESET" >&2; }

banner "install"

# ---------------------------------------------------------------- Python ----
if ! command -v python3 >/dev/null 2>&1; then
    err "python3 is required but was not found."
    exit 1
fi

if ! python3 -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 9) else 1)'; then
    err "Python 3.9 or newer is required (found $(python3 -V 2>&1))."
    exit 1
fi

# ---------------------------------------------------------------- FFmpeg ----
if ! command -v ffmpeg >/dev/null 2>&1; then
    if command -v pacman >/dev/null 2>&1; then
        step "Installing ffmpeg with pacman..."
        if ! sudo pacman -S --needed ffmpeg; then
            err "Could not install ffmpeg. Install it manually and re-run."
            exit 1
        fi
    else
        err "ffmpeg is required. Install it with your package manager and re-run."
        exit 1
    fi
fi

# ------------------------------------------------------- Virtual env -------
step "Creating virtual environment"
note "$VENV"
mkdir -p "$DATA_DIR"
if [ ! -x "$VENV/bin/python" ]; then
    if command -v uv >/dev/null 2>&1; then
        uv venv --python python3 "$VENV"
    else
        python3 -m venv "$VENV"
    fi
fi

step "Installing whisperstt and dependencies"
if command -v uv >/dev/null 2>&1; then
    uv pip install --python "$VENV/bin/python" --upgrade "$SCRIPT_DIR"
else
    "$VENV/bin/python" -m pip install --upgrade pip >/dev/null
    "$VENV/bin/python" -m pip install --upgrade "$SCRIPT_DIR"
fi

# ---------------------------------------------------------- CUDA libs ------
if command -v nvidia-smi >/dev/null 2>&1; then
    if [ -z "$(find "$VENV" -name 'libcublas.so.12*' -print -quit 2>/dev/null)" ]; then
        step "Installing CUDA 12 runtime libraries for ctranslate2"
        if command -v uv >/dev/null 2>&1; then
            uv pip install --python "$VENV/bin/python" \
                nvidia-cublas-cu12 nvidia-cudnn-cu12
        else
            "$VENV/bin/python" -m pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
        fi
        ok "CUDA 12 runtime ready"
    fi
fi

# ------------------------------------------------------------- Command -----
mkdir -p "$BIN_DIR"
ln -sf "$VENV/bin/whisperstt" "$BIN_DIR/whisperstt"
ok "Linked $BIN_DIR/whisperstt"

case ":$PATH:" in
    *":$BIN_DIR:"*) ;;
    *)
        echo
        note "$BIN_DIR is not on your PATH. Add this to your shell config:"
        note "  export PATH=\"\$PATH:$BIN_DIR\""
        ;;
esac

# ------------------------------------------------------ Existing model -----
mkdir -p "$MODELS_DIR"
LEGACY="$SCRIPT_DIR/whisperModels/faster-whisper-large-v3"
if [ -d "$LEGACY" ] && [ ! -d "$MODELS_DIR/large-v3" ]; then
    step "Moving existing model to $MODELS_DIR/large-v3"
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
    step "No model installed yet — choose one:"
    "$VENV/bin/whisperstt" --models
fi

echo
ok "Done."
note "Try: whisperstt /path/to/movie.mkv"
