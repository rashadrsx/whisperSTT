"""Model loading and transcription with a terminal progress bar."""

import sys
import time

from .audio import SAMPLE_RATE, extract_audio
from .cuda import preload_cuda_libraries
from .subtitle import format_clock


def load_model(model_dir, device, compute_type, quiet=False):
    preload_cuda_libraries()
    from faster_whisper import WhisperModel

    if not quiet:
        print("Loading model...")
    return WhisperModel(str(model_dir), device=device, compute_type=compute_type)


def render_progress(done, total, started, width=30):
    done = min(done, total) if total else done
    ratio = min(max(done / total, 0.0), 1.0) if total else 0.0
    filled = int(ratio * width)
    bar = "█" * filled + "░" * (width - filled)
    elapsed = time.perf_counter() - started
    eta = elapsed / ratio - elapsed if ratio else 0
    sys.stderr.write(
        f"\r  {bar} {ratio * 100:5.1f}%  {format_clock(done)} / {format_clock(total)}"
        f"  ETA {format_clock(eta)}"
    )
    sys.stderr.flush()


def transcribe(model, path, language=None, translate=False, vad=True, quiet=False):
    """Transcribe one media file and return its segments."""
    audio = extract_audio(path)
    total_seconds = len(audio) / SAMPLE_RATE

    if not quiet:
        print(f"Transcribing {format_clock(total_seconds)} of audio...")

    options = {"beam_size": 1, "vad_filter": vad}
    if language:
        options["language"] = language
    if translate:
        options["task"] = "translate"

    segments_iter, info = model.transcribe(audio, **options)
    total_seconds = info.duration or total_seconds

    show_progress = not quiet and sys.stderr.isatty()
    started = time.perf_counter()
    segments = []
    for segment in segments_iter:
        segments.append(segment)
        if show_progress:
            render_progress(segment.end, total_seconds, started)
    if show_progress:
        render_progress(total_seconds, total_seconds, started)
        sys.stderr.write("\n")
        sys.stderr.flush()

    return segments
