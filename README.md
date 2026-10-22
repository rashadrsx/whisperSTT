# whisperstt

Transcribe audio and video files into `.srt` subtitles, running on your GPU when
you have one. Powered by [`faster-whisper`](https://github.com/SYSTRAN/faster-whisper).

```console
$ whisperstt Snatch.2000.1080p.BluRay.x264-YTS.AM.mp4

  ▌ whisperstt
    transcribe · large-v3 · cuda
  ────────────────────────────────────────────────────

  › Loading large-v3 · cuda · float16

  ▌ Snatch.2000.1080p.BluRay.x264-YTS.AM.mp4
  › Transcribing 1:42:46 of audio
  █████████████████░░░░░░░░░░░░░  57.9%  59:27 / 1:42:46  ETA 04:02
  ✓ Saved 1674 subtitles → Snatch.2000.1080p.BluRay.x264-YTS.AM.srt
```

## Requirements

* **Python 3.9+**
* **FFmpeg** — used to decode the audio track
* An **NVIDIA GPU** is optional. Without one, transcription runs on the CPU and
  is much slower.

## Install

```bash
git clone <this repository> && cd whisperSTT
./install.sh
```

The installer:

1. installs FFmpeg with `pacman` if it is missing (Arch; other systems are told
   what to install),
2. creates a private virtual environment in `~/.local/share/whisperstt/venv`,
3. installs `whisperstt` and its dependencies into it,
4. installs the CUDA 12 runtime libraries when an NVIDIA GPU is present,
5. links the `whisperstt` command into `~/.local/bin`,
6. adopts a model left over from an older layout, if there is one,
7. opens the model chooser when no model is configured yet.

Re-run `./install.sh` after updating the source to install the new version.

## Usage

```bash
whisperstt movie.mkv                    # one file
whisperstt *.mkv                        # several; the model is loaded once
whisperstt "Episode 01.mkv" -l en       # force a language
whisperstt movie.mkv --translate        # translate speech into English
whisperstt --watch ~/Downloads          # transcribe files as they arrive
```

Subtitles are written next to each input file: `movie.mkv` becomes `movie.srt`.
That is exactly where players such as VLC, mpv, Plex and Jellyfin look for them,
so they load automatically.

| Option | Description |
| --- | --- |
| `-m, --model NAME` | Use a specific model for this run |
| `-l, --language CODE` | Force a language instead of auto-detecting |
| `--translate` | Translate the speech into English |
| `-o, --output DIR` | Write `.srt` files into `DIR` |
| `-f, --force` | Overwrite an existing `.srt` instead of skipping it |
| `--models` | Choose or download a model |
| `--watch DIR` | Watch a directory and transcribe new files |
| `--setup` | Install missing GPU libraries |
| `-q, --quiet` | Only print errors |
| `--device auto\|cuda\|cpu` | Override device detection |

While transcribing, a progress bar with an ETA is drawn on stderr. It is hidden
when the output is redirected, so logs stay clean:

```
  █████████████████░░░░░░░░░░░░░  57.9%  59:27 / 1:42:46  ETA 04:02
```

## Theme

The installer, model chooser, setup screen, progress bar and watcher all share
one warm parchment palette:

| Role | Colour |
| --- | --- |
| Background and light surface | `#FAF9F5` cream |
| Primary text and dark elements | `#141413` slate ink |
| Accent, CTAs and highlights | `#D97757` terracotta |
| Borders and card shading | `#E8E6DC` light greige |
| Secondary text and boundaries | `#B1ADA1` warm slate |

Colour is emitted only to a terminal. It is disabled by `NO_COLOR`,
`CLICOLOR=0` or `TERM=dumb`, and 256-colour terminals receive approximate
fallbacks. Set `FORCE_COLOR=1` to keep the palette when output is piped.
Text automatically switches to slate ink on light terminals (using
`COLORFGBG`); set `WHISPERSTT_BACKGROUND=light` or `dark` to force it.

## Models

Run `whisperstt --models` to change the default. Sizes and VRAM figures are
for the packaged `faster-whisper` models:

| Model | Disk | VRAM (float16) | Notes |
| --- | --- | --- | --- |
| `tiny` | 75 MB | ~0.8 GB | fastest, lowest accuracy |
| `base` | 141 MB | ~1.0 GB | |
| `small` | 464 MB | ~1.5 GB | good speed/accuracy balance |
| `medium` | 1.4 GB | ~2.9 GB | |
| `large-v3-turbo` | 1.5 GB | ~2.9 GB | near `large-v3` quality, ~3x faster |
| `large-v3` | 2.9 GB | ~4.9 GB | best accuracy |

The VRAM column includes roughly 0.6 GB of CUDA overhead. If a model does not
fit in your GPU's memory, a quantised `int8_float16` version is used
automatically, which roughly halves the requirement.

Models are stored in `~/.local/share/whisperstt/models/<model>` and are
downloaded on first use, from the Hugging Face Hub.

## Files and configuration

| What | Where |
| --- | --- |
| Command | `~/.local/bin/whisperstt` |
| Virtualenv | `~/.local/share/whisperstt/venv` |
| Models | `~/.local/share/whisperstt/models` |
| Settings | `~/.config/whisperstt/config.json` |

`XDG_DATA_HOME`, `XDG_CONFIG_HOME` and `XDG_BIN_HOME` are respected. The
settings file is written by `whisperstt --models`; the available keys are:

```json
{
  "model": "large-v3",
  "language": null,
  "output_dir": null,
  "device": "auto",
  "vad": true,
  "compute_type": "auto"
}
```

## Notes

* **Silence and music** are skipped by voice activity detection (`vad`). Without
  it, Whisper invents lines during music or long pauses, and may misdetect the
  language of a film that opens with a score.
* **`--translate` only translates into English.** Whisper cannot translate into
  other languages.
* **A system CUDA 13 install is not enough** on its own: `ctranslate2` is built
  against CUDA 12 and looks for `libcublas.so.12`. The installer adds the CUDA 12
  runtime as pip packages and `whisperstt` preloads them, so no
  `LD_LIBRARY_PATH` tweaking is needed.
* **Existing `.srt` files are never overwritten** unless you pass `--force`.

## Development

```bash
python3 -m venv venv
venv/bin/python -m pip install -e .
venv/bin/python -m unittest discover -s tests -v
```

The code lives in `src/whisperstt`:

| Module | Responsibility |
| --- | --- |
| `cli.py` | Argument parsing, file planning, top-level flow |
| `theme.py` | Palette, colour detection and shared UI glyphs |
| `config.py` | XDG paths and `config.json` |
| `models.py` | Model table, VRAM sizing, chooser and downloader |
| `cuda.py` | GPU probing and CUDA 12 library preloading |
| `audio.py` | FFmpeg decoding to 16 kHz mono samples |
| `subtitle.py` | SRT formatting and atomic file writing |
| `transcribe.py` | Model loading, transcription, progress bar |
| `watch.py` | Directory watching |
| `setup.py` | `--setup` dependency installation |
