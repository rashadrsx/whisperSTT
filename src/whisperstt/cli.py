"""Command line interface for whisperstt."""

import argparse
import sys
from pathlib import Path

from . import __version__, config, models
from .cuda import detect_device, detect_gpu
from .subtitle import write_srt
from .transcribe import load_model, transcribe


def build_parser():
    parser = argparse.ArgumentParser(
        prog="whisperstt",
        description="Transcribe audio and video files into .srt subtitles.",
        epilog="Run 'whisperstt --models' to choose or download a model.",
    )
    parser.add_argument("files", nargs="*", metavar="FILE", help="audio or video files")
    parser.add_argument("-m", "--model", help="model to use for this run")
    parser.add_argument("-l", "--language", help="force a language (default: auto)")
    parser.add_argument(
        "--translate", action="store_true", help="translate speech into English"
    )
    parser.add_argument("-o", "--output", metavar="DIR", help="write .srt files to DIR")
    parser.add_argument(
        "-f", "--force", action="store_true", help="overwrite existing .srt files"
    )
    parser.add_argument(
        "--models", action="store_true", help="choose or download a model"
    )
    parser.add_argument(
        "--setup", action="store_true", help="install missing GPU libraries"
    )
    parser.add_argument(
        "--watch", metavar="DIR", help="watch DIR and transcribe new files"
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="only print errors")
    parser.add_argument(
        "--device", choices=("auto", "cuda", "cpu"), help="override device detection"
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def output_path(source, output_dir):
    """Return the .srt path for a media file (beside it by default)."""
    name = source.stem + ".srt"
    if output_dir:
        return Path(output_dir).expanduser() / name
    return source.with_suffix(".srt")


def resolve_model(args, settings):
    """Return the model name, asking and downloading when necessary."""
    name = args.model or settings.get("model")

    if not name:
        if not sys.stdin.isatty():
            raise SystemExit(
                "No model configured. Run 'whisperstt --models' to download one, "
                "or pass --model NAME."
            )
        name = models.choose_model()
        settings["model"] = name
        config.save(settings)
    elif not models.is_installed(name):
        models.ensure_model(name, quiet=args.quiet)

    return name


def plan_files(files, args, settings):
    """Split inputs into work to do and problems, before loading any model."""
    pending = []
    failures = 0
    for name in files:
        source = Path(name).expanduser()
        if not source.is_file():
            print(f"Not found: {source}", file=sys.stderr)
            failures += 1
            continue
        target = output_path(source, args.output or settings.get("output_dir"))
        if target.exists() and not args.force:
            print(f"Skipping {source.name}: {target.name} already exists (use --force)")
            continue
        pending.append((source, target))
    return pending, failures


def transcribe_one(model, source, target, args, settings):
    """Transcribe a single file. Returns False on failure."""
    if not args.quiet:
        print(f"\n{source.name}")
    try:
        segments = transcribe(
            model,
            source,
            language=args.language or settings.get("language"),
            translate=args.translate,
            vad=settings.get("vad", True),
            quiet=args.quiet,
        )
    except Exception as exc:
        print(f"Failed: {source.name}: {exc}", file=sys.stderr)
        return False

    target.parent.mkdir(parents=True, exist_ok=True)
    count = write_srt(target, segments)
    if not args.quiet:
        print(f"Saved {count} subtitles -> {target}")
    return True


def main(argv=None):
    # Keep status output visible in real time when redirected to a file or pipe.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    parser = build_parser()
    args = parser.parse_args(argv)
    settings = config.load()

    if args.setup:
        from .setup import run_setup

        return run_setup()

    if args.models:
        name = models.choose_model(settings.get("model"))
        models.ensure_model(name, quiet=args.quiet)
        settings["model"] = name
        config.save(settings)
        if not args.quiet:
            print(f"Default model: {name}")
        return 0

    if not args.files and not args.watch:
        parser.print_help()
        return 2

    pending, failures = plan_files(args.files, args, settings)
    if not pending and not args.watch:
        return 1 if failures else 0

    name = resolve_model(args, settings)
    entry = models.get(name)

    device = args.device or settings.get("device", "auto")
    if device == "auto":
        device = detect_device()
    gpu = detect_gpu() if device == "cuda" else None
    compute_type = settings.get("compute_type", "auto")
    if compute_type == "auto":
        compute_type = models.compute_type_for(
            entry, device, gpu[1] if gpu else None
        )

    model = load_model(
        models.model_path(name), device, compute_type, quiet=args.quiet
    )

    for source, target in pending:
        if not transcribe_one(model, source, target, args, settings):
            failures += 1

    if args.watch:
        from .watch import watch_directory

        def on_new(path):
            target = output_path(path, args.output or settings.get("output_dir"))
            if target.exists() and not args.force:
                print(f"Skipping {path.name}: {target.name} already exists")
                return
            transcribe_one(model, path, target, args, settings)

        watch_directory(args.watch, on_new)

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
