"""Model loading and transcription with a terminal progress bar."""

import sys
import time

import numpy as np

from .audio import SAMPLE_RATE, extract_audio
from .cuda import preload_cuda_libraries
from .subtitle import format_clock

# faster-whisper turns the whole waveform into a mel spectrogram in one go, so
# memory grows with the length of the file (~3 GB per hour of audio). Feeding it
# bounded windows keeps memory flat, so long audiobooks don't get OOM-killed.
CHUNK_SECONDS = 20 * 60
SPLIT_SEARCH_SECONDS = 20
SPLIT_FRAME_SECONDS = 0.5


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
        print(f"Transcribing {format_clock(total_seconds)} of audio...")

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
