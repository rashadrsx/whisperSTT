"""Dependency checks used by `whisperstt --setup`."""

import shutil
import subprocess
import sys

from . import theme
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
        print(
            theme.paint(
                "No pip or uv available to install packages.",
                "accent",
                "bold",
                stream=sys.stderr,
            ),
            file=sys.stderr,
        )
        return False
    try:
        subprocess.run([*command, *packages], check=True)
        return True
    except (subprocess.CalledProcessError, OSError):
        return False


def run_setup():
    problems = []
    print(theme.banner("setup"))

    if shutil.which("ffmpeg") is None:
        problems.append("ffmpeg is missing")
        print(
            f"  {theme.mark('fail')} {theme.paint('ffmpeg', 'primary', 'bold')} "
            f"{theme.paint('missing', 'accent', 'bold')}"
        )
        print(
            f"    {theme.paint('Install it with your package manager ', 'muted')}"
            f"{theme.paint('(Arch: sudo pacman -S ffmpeg).', 'muted')}"
        )
    else:
        print(f"  {theme.mark('ok')} {theme.paint('ffmpeg', 'primary')}")

    device = detect_device()
    gpu = detect_gpu()
    if device == "cuda" and gpu:
        print(
            f"  {theme.mark('ok')} {theme.paint(gpu[0], 'primary', 'bold')} "
            f"{theme.paint(f'· {gpu[1]} MB VRAM', 'muted')}"
        )
        print(
            f"  {theme.mark('step')} "
            f"{theme.paint('Installing CUDA 12 runtime libraries...', 'primary')}"
        )
        if _install(list(GPU_PACKAGES)):
            print(
                f"  {theme.mark('ok')} "
                f"{theme.paint('CUDA 12 runtime ready', 'primary')}"
            )
        else:
            problems.append(
                "could not install the CUDA 12 libraries; run: "
                f"{sys.executable} -m pip install {' '.join(GPU_PACKAGES)}"
            )
            message = "CUDA 12 runtime could not be installed"
            print(
                f"  {theme.mark('fail')} "
                f"{theme.paint(message, 'accent', 'bold')}"
            )
    else:
        print(
            f"  {theme.mark('bullet')} "
            f"{theme.paint('No CUDA GPU detected', 'primary')} "
            f"{theme.paint('· transcription will use the CPU', 'muted')}"
        )

    if problems:
        for problem in problems:
            print(f"  {theme.mark('warn')} {theme.paint(problem, 'accent')}")
        return 1

    print(
        f"  {theme.mark('ok')} {theme.paint('Setup complete', 'primary', 'bold')} "
        f"{theme.paint('· try: whisperstt movie.mkv', 'muted')}"
    )
    return 0
