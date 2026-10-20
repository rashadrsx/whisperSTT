"""Watch a directory and transcribe files as they appear."""

import time
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

STABLE_CHECKS = 3
STABLE_INTERVAL = 1.0


def _wait_until_stable(path, checks=STABLE_CHECKS, interval=STABLE_INTERVAL):
    """Wait until a freshly copied file stops growing."""
    previous = -1
    stable = 0
    for _ in range(120):
        try:
            size = path.stat().st_size
        except OSError:
            return False
        if size and size == previous:
            stable += 1
            if stable >= checks:
                return True
        else:
            stable = 0
        previous = size
        time.sleep(interval)
    return False


class _Handler(FileSystemEventHandler):
    def __init__(self, callback, ignore):
        self.callback = callback
        self.ignore = ignore

    def on_created(self, event):
        if event.is_directory:
            return
        path = Path(event.src_path)
        if path.suffix.lower() in self.ignore:
            return
        if not _wait_until_stable(path):
            print(f"Gave up waiting for {path.name} to finish copying.")
            return
        self.callback(path)


def watch_directory(directory, callback, ignore_suffixes=(".srt", ".tmp", ".part")):
    directory = Path(directory).expanduser()
    directory.mkdir(parents=True, exist_ok=True)

    observer = Observer()
    observer.schedule(_Handler(callback, ignore_suffixes), str(directory), recursive=False)
    observer.start()
    print(f"Watching {directory} — press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping watcher...")
        observer.stop()
    observer.join()
