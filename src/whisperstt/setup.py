"""Dependency checks used by `whisperstt --setup`."""

import shutil
import subprocess
import sys

from .cuda import detect_device, detect_gpu

GPU_PACKAGES = ("nvidia-cublas-cu12", "nvidia-cudnn-cu12")


def _install_command():
    """Return the working install command prefix, or None."""
    pip = [sys.executable, "-m", "pip", "install"]
    probe = subprocess.run(
        [sys.executable, "-m", "pip", "--version"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if probe.returncode == 0:
        return pip
    if shutil.which("uv"):
        return ["uv", "pip", "install", "--python", sys.executable]
    return None


def _install(packages):
    """Install packages into the running interpreter's environment."""
    command = _install_command()
    if command is None:
        print("No pip or uv available to install packages.", file=sys.stderr)
        return False
    try:
        subprocess.run([*command, *packages], check=True)
        return True
    except (subprocess.CalledProcessError, OSError):
        return False


def run_setup():
    problems = []

    if shutil.which("ffmpeg") is None:
        problems.append("ffmpeg is missing: install it (on Arch: sudo pacman -S ffmpeg)")

    device = detect_device()
    gpu = detect_gpu()
    if device == "cuda" and gpu:
        print(f"GPU: {gpu[0]} ({gpu[1]} MB VRAM)")
        print("Installing CUDA 12 runtime libraries...")
        if not _install(list(GPU_PACKAGES)):
            problems.append(
                "could not install the CUDA 12 libraries; run: "
                f"{sys.executable} -m pip install {' '.join(GPU_PACKAGES)}"
            )
    else:
        print("No CUDA GPU detected: transcription will use the CPU.")

    if problems:
        for problem in problems:
            print(f"  ! {problem}")
        return 1

    print("Setup complete.")
    return 0
