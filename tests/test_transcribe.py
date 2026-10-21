import sys
import unittest
from types import SimpleNamespace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from whisperstt import transcribe

SAMPLE_RATE = transcribe.SAMPLE_RATE


class FakeSegment:
    def __init__(self, start, end, text="hello"):
        self.start = start
        self.end = end
        self.text = text


class FakeModel:
    """Stand-in for WhisperModel that records the chunks it receives."""

    def __init__(self):
        self.calls = []
        self.languages = iter(["en", "fr", "es"])

    def transcribe(self, audio, **options):
        self.calls.append((len(audio), options))
        info = SimpleNamespace(language=next(self.languages, "en"))
        return iter([FakeSegment(0.0, 1.0)]), info


class SplitPointTest(unittest.TestCase):
    def test_cuts_at_quietest_frame(self):
        audio = np.full(100 * SAMPLE_RATE, 0.25, dtype=np.float32)
        quiet = 46 * SAMPLE_RATE
        audio[quiet : quiet + SAMPLE_RATE // 2] = 0.0

        point = transcribe.split_point(audio, 50 * SAMPLE_RATE)

        self.assertEqual(point, quiet)

    def test_returns_target_when_search_window_is_too_small(self):
        audio = np.zeros(100, dtype=np.float32)
        self.assertEqual(transcribe.split_point(audio, 50 * 16), 50 * 16)


class TranscribeAudioTest(unittest.TestCase):
    def test_splits_long_audio_and_offsets_timestamps(self):
        model = FakeModel()
        length = 2 * transcribe.CHUNK_SECONDS * SAMPLE_RATE + 5 * SAMPLE_RATE

        segments = transcribe.transcribe_audio(
            model, np.zeros(length, dtype=np.float32), quiet=True
        )

        self.assertEqual(len(model.calls), 3)
        self.assertTrue(all(size <= transcribe.CHUNK_SECONDS * SAMPLE_RATE for size, _ in model.calls))
        self.assertEqual([round(segment.start) for segment in segments], [0, 1180, 2360])

    def test_detects_language_once_and_reuses_it(self):
        model = FakeModel()
        length = 2 * transcribe.CHUNK_SECONDS * SAMPLE_RATE + 5 * SAMPLE_RATE

        transcribe.transcribe_audio(model, np.zeros(length, dtype=np.float32), quiet=True)

        self.assertNotIn("language", model.calls[0][1])
        self.assertEqual(model.calls[1][1]["language"], "en")
        self.assertEqual(model.calls[2][1]["language"], "en")

    def test_forced_language_is_used_for_every_chunk(self):
        model = FakeModel()
        length = 2 * transcribe.CHUNK_SECONDS * SAMPLE_RATE + 5 * SAMPLE_RATE

        transcribe.transcribe_audio(
            model, np.zeros(length, dtype=np.float32), language="de", quiet=True
        )

        self.assertEqual([options["language"] for _, options in model.calls], ["de", "de", "de"])


if __name__ == "__main__":
    unittest.main()
