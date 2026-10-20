"""SubRip subtitle formatting."""

import os


def format_srt_time(seconds):
    """Convert seconds into the SRT timestamp format: HH:MM:SS,mmm."""
    milliseconds = int(round(max(seconds, 0.0) * 1000))
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    secs, milliseconds = divmod(milliseconds, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def format_clock(seconds):
    """Format a duration as mm:ss, or h:mm:ss when it is an hour or longer."""
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def write_srt(path, segments):
    """Write segments as an SRT file, atomically. Returns the cue count."""
    path = str(path)
    temporary = path + ".tmp"
    index = 1
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            for segment in segments:
                text = segment.text.strip()
                if not text:
                    continue
                handle.write(f"{index}\n")
                handle.write(
                    f"{format_srt_time(segment.start)} --> "
                    f"{format_srt_time(segment.end)}\n"
                )
                handle.write(f"{text}\n\n")
                index += 1
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.remove(temporary)
        raise
    return index - 1
