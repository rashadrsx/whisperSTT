"""CUDA helpers: library preloading and GPU probing."""

import ctypes
import glob
import os
import subprocess
import sysconfig


def preload_cuda_libraries():
    """Load the CUDA 12 runtime libraries shipped in the nvidia-* wheels.

    ctranslate2 is built against CUDA 12 and dlopen()s ``libcublas.so.12`` /
    ``libcudnn.so.9`` by their versioned names, so a system CUDA 13 (which only
    provides ``libcublas.so.13``) is not enough. Preloading the ones from the
    venv makes the GPU work without having to export LD_LIBRARY_PATH.
    """
    site_packages = sysconfig.get_paths()["purelib"]
    patterns = (
        os.path.join(site_packages, "nvidia", "*", "lib", "*.so*"),
        os.path.join(site_packages, "nvidia", "*", "bin", "*.dll"),
    )
    for pattern in patterns:
        for library in sorted(glob.glob(pattern)):
            try:
                ctypes.CDLL(library, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass
    if hasattr(os, "add_dll_directory"):
        for directory in glob.glob(os.path.join(site_packages, "nvidia", "*", "bin")):
            try:
                os.add_dll_directory(directory)
            except OSError:
                pass


def _first_line(command):
    try:
        output = subprocess.check_output(
            command, stderr=subprocess.DEVNULL, text=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return output.strip().splitlines()[0] if output.strip() else None


def detect_device():
    """Return the best available device name: "cuda" or "cpu"."""
    if _first_line(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]):
        return "cuda"
    if _first_line(["rocm-smi", "--showproductname"]):
        return "cuda"  # faster-whisper uses "cuda" for ROCm
    return "cpu"


def detect_gpu():
    """Return (name, vram_mb) for the first NVIDIA GPU, or None."""
    line = _first_line(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"]
    )
    if not line:
        return None
    name, _, vram = line.rpartition(",")
    try:
        return name.strip(), int(float(vram))
    except ValueError:
        return None
