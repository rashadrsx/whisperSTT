import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from whisperstt import models


class ModelTableTest(unittest.TestCase):
    def test_get_returns_entry(self):
        self.assertEqual(models.get("large-v3")["repo"], "Systran/faster-whisper-large-v3")

    def test_unknown_model_exits(self):
        with self.assertRaises(SystemExit):
            models.get("nonexistent")

    def test_format_size_switches_to_gigabytes(self):
        self.assertEqual(models.format_size(464), "464 MB")
        self.assertEqual(models.format_size(2948), "2.9 GB")

    def test_default_model_is_largest_that_fits(self):
        self.assertEqual(models.default_model(None), "small")
        self.assertEqual(models.default_model(("GPU", 2048)), "small")
        self.assertEqual(models.default_model(("GPU", 6144)), "large-v3")

    def test_compute_type_prefers_float16_when_it_fits(self):
        entry = models.get("large-v3")
        self.assertEqual(models.compute_type_for(entry, "cuda", 6144), "float16")
        self.assertEqual(models.compute_type_for(entry, "cuda", 4096), "int8_float16")
        self.assertEqual(models.compute_type_for(entry, "cpu", None), "int8")

    def test_is_installed_checks_model_bin(self):
        with tempfile.TemporaryDirectory() as tmp:
            original = models.config.models_dir
            models.config.models_dir = lambda: Path(tmp)
            try:
                self.assertFalse(models.is_installed("small"))
                target = Path(tmp) / "small"
                target.mkdir()
                (target / "model.bin").write_bytes(b"x")
                self.assertTrue(models.is_installed("small"))
            finally:
                models.config.models_dir = original


if __name__ == "__main__":
    unittest.main()
