import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from whisperstt.subtitle import format_clock, format_srt_time, write_srt


class Segment:
    def __init__(self, start, end, text):
        self.start = start
        self.end = end
        self.text = text


class FormatSrtTimeTest(unittest.TestCase):
    def test_zero(self):
        self.assertEqual(format_srt_time(0), "00:00:00,000")

    def test_fractional_rounds_to_milliseconds(self):
        self.assertEqual(format_srt_time(1.5), "00:00:01,500")
        self.assertEqual(format_srt_time(61.25), "00:01:01,250")

    def test_beyond_one_hour(self):
        self.assertEqual(format_srt_time(3725.4), "01:02:05,400")

    def test_negative_is_clamped(self):
        self.assertEqual(format_srt_time(-3), "00:00:00,000")


class FormatClockTest(unittest.TestCase):
    def test_under_an_hour(self):
        self.assertEqual(format_clock(180), "03:00")

    def test_over_an_hour(self):
        self.assertEqual(format_clock(6166), "1:42:46")


class WriteSrtTest(unittest.TestCase):
    def test_writes_valid_blocks_and_skips_blank_text(self):
        segments = [
            Segment(0.0, 1.5, "Hello"),
            Segment(1.5, 2.0, "   "),
            Segment(2.0, 3.25, " World "),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "movie.srt"
            count = write_srt(target, segments)
            self.assertEqual(count, 2)
            self.assertEqual(
                target.read_text(encoding="utf-8"),
                "1\n00:00:00,000 --> 00:00:01,500\nHello\n\n"
                "2\n00:00:02,000 --> 00:00:03,250\nWorld\n\n",
            )
            self.assertFalse(Path(str(target) + ".tmp").exists())

    def test_replaces_existing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "movie.srt"
            target.write_text("old", encoding="utf-8")
            write_srt(target, [Segment(0, 1, "new")])
            self.assertIn("new", target.read_text(encoding="utf-8"))
            self.assertNotIn("old", target.read_text(encoding="utf-8"))

    def test_cleans_up_temporary_file_on_failure(self):
        class Broken:
            start, end, text = 0, 1, "boom"

            def __getattribute__(self, name):
                raise RuntimeError("boom")

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "movie.srt"
            with self.assertRaises(RuntimeError):
                write_srt(target, [Broken()])
            self.assertFalse(Path(str(target) + ".tmp").exists())
            self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
