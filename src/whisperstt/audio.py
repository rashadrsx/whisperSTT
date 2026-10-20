"""Audio decoding via ffmpeg."""

import subprocess

import numpy as np

SAMPLE_RATE = 16000


def extract_audio(path):
    """Decode the audio track of a media file to mono float32 samples."""
    command = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-ar",
        str(SAMPLE_RATE),
        "-ac",
        "1",
        "-f",
        "s16le",
        "-",
    ]
    try:
        with subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        ) as process:
            audio_bytes, error = process.communicate()
    except FileNotFoundError:
        raise SystemExit("ffmpeg is not installed or not on your PATH.")

    if process.returncode != 0 or not audio_bytes:
        message = error.decode(errors="replace").strip() or "no audio stream found"
        raise SystemExit(f"Could not decode audio from {path}:\n{message}")

    return np.frombuffer(audio_bytes, np.int16).astype(np.float32) / 32768.0
