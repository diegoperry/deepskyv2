from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from astropy.io import fits

from app.data_score import evaluate_data_score
from app.web_app import process_page


class DataScoreTests(unittest.TestCase):
    @staticmethod
    def _astro_field(seed: int, noise: float) -> np.ndarray:
        rng = np.random.default_rng(seed)
        height, width = 320, 420
        yy, xx = np.mgrid[:height, :width]
        image = np.full((height, width), 0.025, dtype=np.float32)
        image += 0.24 * np.exp(-(((xx - 205) / 65) ** 2 + ((yy - 165) / 38) ** 2))
        for _ in range(80):
            x = rng.integers(4, width - 4)
            y = rng.integers(4, height - 4)
            image[y - 1:y + 2, x - 1:x + 2] += rng.uniform(0.08, 0.45)
        image += rng.normal(0.0, noise, image.shape).astype(np.float32)
        return np.clip(image, 0.0, 1.0)

    def test_cleaner_capture_scores_higher_than_noisy_capture(self) -> None:
        with TemporaryDirectory() as folder:
            root = Path(folder)
            clean_path = root / "clean.fit"
            noisy_path = root / "noisy.fit"
            fits.PrimaryHDU(self._astro_field(7, 0.004)).writeto(clean_path)
            fits.PrimaryHDU(self._astro_field(7, 0.035)).writeto(noisy_path)

            clean = evaluate_data_score(clean_path)
            noisy = evaluate_data_score(noisy_path)

            self.assertGreater(clean["score"], noisy["score"])
            self.assertTrue(1 <= noisy["score"] <= 100)
            self.assertTrue(1 <= clean["score"] <= 100)

    def test_fits_stack_metadata_produces_time_recommendation(self) -> None:
        with TemporaryDirectory() as folder:
            path = Path(folder) / "stack.fit"
            hdu = fits.PrimaryHDU(self._astro_field(11, 0.025))
            hdu.header["EXPTIME"] = 10.0
            hdu.header["STACKCNT"] = 20
            hdu.writeto(path)

            result = evaluate_data_score(path)

            self.assertEqual(result["total_exposure_seconds"], 200.0)
            self.assertIn("capture", result["recommendation"].lower())
            self.assertIsInstance(result["factors"], list)

    def test_process_page_includes_upload_score_panel(self) -> None:
        html = process_page()
        self.assertIn('id="dataScorePanel"', html)
        self.assertIn('id="dataScoreValue"', html)
        self.assertIn("renderDataScore(preview.data_score)", html)
        self.assertIn("Capture Data Score", html)


if __name__ == "__main__":
    unittest.main()

