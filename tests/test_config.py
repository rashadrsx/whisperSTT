import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from whisperstt import config


class ConfigPathsTest(unittest.TestCase):
    def test_respects_xdg_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = {"XDG_CONFIG_HOME": tmp + "/cfg", "XDG_DATA_HOME": tmp + "/data"}
            with mock.patch.dict(os.environ, env):
                self.assertEqual(config.config_file(), Path(tmp) / "cfg/whisperstt/config.json")
                self.assertEqual(config.models_dir(), Path(tmp) / "data/whisperstt/models")
                self.assertEqual(config.venv_dir(), Path(tmp) / "data/whisperstt/venv")

    def test_falls_back_to_home(self):
        with mock.patch.dict(os.environ, {"HOME": "/home/tester"}, clear=False):
            for key in ("XDG_CONFIG_HOME", "XDG_DATA_HOME"):
                os.environ.pop(key, None)
            self.assertEqual(
                config.config_file(), Path("/home/tester/.config/whisperstt/config.json")
            )
            self.assertEqual(
                config.models_dir(), Path("/home/tester/.local/share/whisperstt/models")
            )


class ConfigFileTest(unittest.TestCase):
    def test_load_returns_defaults_when_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, {"XDG_CONFIG_HOME": tmp}):
                settings = config.load()
                self.assertIsNone(settings["model"])
                self.assertTrue(settings["vad"])

    def test_save_and_load_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, {"XDG_CONFIG_HOME": tmp}):
                config.save({"model": "small"})
                self.assertEqual(config.load()["model"], "small")

    def test_corrupt_file_is_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, {"XDG_CONFIG_HOME": tmp}):
                path = config.config_file()
                path.parent.mkdir(parents=True)
                path.write_text("{not json", encoding="utf-8")
                self.assertIsNone(config.load()["model"])


if __name__ == "__main__":
    unittest.main()
