import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from whisperstt import theme


class FakeTty:
    def isatty(self):
        return True


class FakePipe:
    def isatty(self):
        return False


class PaintTest(unittest.TestCase):
    def test_pipes_get_plain_text(self):
        with mock.patch.dict(os.environ, {"TERM": "xterm-256color"}, clear=True):
            self.assertEqual(theme.paint("hello", "accent", stream=FakePipe()), "hello")

    def test_truecolor_uses_palette(self):
        env = {"TERM": "xterm-256color", "COLORTERM": "truecolor"}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(
                theme.paint("hello", "accent", stream=FakeTty()),
                "\033[38;2;217;119;87mhello\033[0m",
            )

    def test_no_color_wins_over_force(self):
        env = {"TERM": "xterm-256color", "NO_COLOR": "1", "FORCE_COLOR": "1"}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(theme.paint("hello", "accent", stream=FakeTty()), "hello")

    def test_force_color_styles_pipes(self):
        env = {"TERM": "xterm-256color", "FORCE_COLOR": "1"}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertIn("\033[", theme.paint("hello", "accent", stream=FakePipe()))

    def test_light_terminal_uses_slate_ink_for_primary(self):
        env = {
            "TERM": "xterm-256color",
            "COLORTERM": "truecolor",
            "COLORFGBG": "0;15",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(
                theme.paint("hello", "primary", stream=FakeTty()),
                "\033[38;2;20;20;19mhello\033[0m",
            )

    def test_background_override_beats_colorfgbg(self):
        env = {
            "TERM": "xterm-256color",
            "COLORTERM": "truecolor",
            "COLORFGBG": "15;0",
            "WHISPERSTT_BACKGROUND": "light",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(
                theme.paint("hello", "primary", stream=FakeTty()),
                "\033[38;2;20;20;19mhello\033[0m",
            )

    def test_256_colour_fallback(self):
        env = {"TERM": "xterm-256color"}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(
                theme.paint("hello", "terracotta", stream=FakeTty()),
                "\033[38;5;173mhello\033[0m",
            )

    def test_background_role(self):
        env = {"TERM": "xterm-256color", "COLORTERM": "truecolor"}
        with mock.patch.dict(os.environ, env, clear=True):
            styled = theme.paint(" x ", "bg:terracotta", stream=FakeTty())
            self.assertIn("\033[48;2;217;119;87m", styled)

    def test_visible_len_ignores_escape_sequences(self):
        env = {"TERM": "xterm-256color", "COLORTERM": "truecolor"}
        with mock.patch.dict(os.environ, env, clear=True):
            styled = theme.paint("hello", "accent", stream=FakeTty())
        self.assertEqual(theme.visible_len(styled), 5)

    def test_banner_and_marks(self):
        env = {"TERM": "xterm-256color"}
        with mock.patch.dict(os.environ, env, clear=True):
            banner = theme.banner(stream=FakePipe())
        self.assertIn("whisperstt", banner)
        self.assertIn("books · movies · episodes", banner)
        self.assertEqual(theme.mark("ok", stream=FakePipe()), "✓")


if __name__ == "__main__":
    unittest.main()
