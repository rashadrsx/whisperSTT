"""Model loading and transcription with a terminal progress bar."""

import sys
import time

import numpy as np

from . import theme
from .audio import SAMPLE_RATE, extract_audio
from .cuda import preload_cuda_libraries
from .subtitle import format_clock

# faster-whisper turns the whole waveform into a mel spectrogram in one go, so
# memory grows with the length of the file (~3 GB per hour of audio). Feeding it
# bounded windows keeps memory flat, so long audiobooks don't get OOM-killed.
CHUNK_SECONDS = 20 * 60
SPLIT_SEARCH_SECONDS = 20
SPLIT_FRAME_SECONDS = 0.5


def load_model(model_dir, device, compute_type, quiet=False, name=None):
    preload_cuda_libraries()
    from faster_whisper import WhisperModel

    if not quiet:
        label = name or getattr(model_dir, "name", str(model_dir))
        print(
            f"  {theme.mark('step')} {theme.paint('Loading', 'primary')} "
            f"{theme.paint(label, 'accent', 'bold')} "
            f"{theme.paint(f'· {device} · {compute_type}', 'muted')}"
        )
    return WhisperModel(str(model_dir), device=device, compute_type=compute_type)


def render_progress(done, total, started, width=30):
    done = min(done, total) if total else done
    ratio = min(max(done / total, 0.0), 1.0) if total else 0.0
    filled = int(ratio * width)
    bar = theme.paint("█" * filled, "accent", stream=sys.stderr) + theme.paint(
        "░" * (width - filled), "muted", stream=sys.stderr
    )
    elapsed = time.perf_counter() - started
    eta = elapsed / ratio - elapsed if ratio else 0
    percent = theme.paint(
        f"{ratio * 100:5.1f}%", "accent", "bold", stream=sys.stderr
    )
    clocks = theme.paint(
        f"{format_clock(done)} / {format_clock(total)}",
        "primary",
        stream=sys.stderr,
    )
    remaining = theme.paint(
        f"ETA {format_clock(eta)}", "muted", stream=sys.stderr
    )
    sys.stderr.write(f"\r  {bar} {percent}  {clocks}  {remaining}")
    sys.stderr.flush()


def split_point(audio, target):
    """Return a sample index near ``target`` where the audio is quietest.

    Cutting on silence avoids losing a word at the seam between windows.
    """
    frame = int(SPLIT_FRAME_SECONDS * SAMPLE_RATE)
    reach = SPLIT_SEARCH_SECONDS * SAMPLE_RATE
    start = max(0, target - reach)
    end = min(len(audio), target + reach)
    frames = (end - start) // frame
    if frames < 2:
        return target
    window = audio[start : start + frames * frame].reshape(frames, frame)
    energy = np.abs(window).mean(axis=1)
    return start + int(np.argmin(energy)) * frame


def transcribe(model, path, language=None, translate=False, vad=True, quiet=False):
    """Transcribe one media file and return its segments."""
    return transcribe_audio(
        model,
        extract_audio(path),
        language=language,
        translate=translate,
        vad=vad,
        quiet=quiet,
    )


def transcribe_audio(model, audio, language=None, translate=False, vad=True, quiet=False):
    """Transcribe decoded audio in memory, returning segments with file timestamps."""
    total_seconds = len(audio) / SAMPLE_RATE

    if not quiet:
        print(
            f"  {theme.mark('step')} {theme.paint('Transcribing', 'primary')} "
            f"{theme.paint(format_clock(total_seconds), 'accent', 'bold')} "
            f"{theme.paint('of audio', 'muted')}"
        )

    options = {"beam_size": 1, "vad_filter": vad}
    if translate:
        options["task"] = "translate"

    show_progress = not quiet and sys.stderr.isatty()
    started = time.perf_counter()
    segments = []
    detected = language
    chunk_samples = CHUNK_SECONDS * SAMPLE_RATE
    offset = 0

    while offset < len(audio):
        end = len(audio)
        if end - offset > chunk_samples:
            end = split_point(audio, offset + chunk_samples)
            end = max(end, offset + 1)  # guarantee forward progress

        chunk_options = dict(options)
        if detected:
            # Detect the language once, then reuse it for the remaining windows
            # so a noisy window cannot switch language mid-file.
            chunk_options["language"] = detected
        chunk_segments, info = model.transcribe(audio[offset:end], **chunk_options)
        if not detected:
            detected = info.language

        shift = offset / SAMPLE_RATE
        for segment in chunk_segments:
            segment.start += shift
            segment.end += shift
            segments.append(segment)
            if show_progress:
                render_progress(segment.end, total_seconds, started)
        offset = end

    if show_progress:
        render_progress(total_seconds, total_seconds, started)
        sys.stderr.write("\n")
        sys.stderr.flush()

    return segments
