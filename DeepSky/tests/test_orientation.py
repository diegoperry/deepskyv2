from __future__ import annotations

import unittest

import numpy as np

from app.pipeline import _orient_like_reference as orient_current_pipeline_like_reference
from app.web_legacy_150_pipeline import _orient_like_reference as orient_web_pipeline_like_reference


class PipelineOrientationTests(unittest.TestCase):
    def test_pipeline_corrects_only_high_confidence_orientation_flips(self) -> None:
        height, width = 180, 260
        yy, xx = np.mgrid[:height, :width].astype(np.float32)
        reference_lum = (
            0.025
            + np.exp(-(((xx - 61.0) / 22.0) ** 2 + ((yy - 47.0) / 31.0) ** 2)) * 0.52
            + np.exp(-(((xx - 194.0) / 35.0) ** 2 + ((yy - 126.0) / 18.0) ** 2)) * 0.27
        )
        reference = np.stack(
            [reference_lum * 1.12, reference_lum * 0.94, reference_lum * 0.81],
            axis=2,
        ).astype(np.float32)
        flipped = np.flipud(reference) ** 0.91

        for orient in (orient_web_pipeline_like_reference, orient_current_pipeline_like_reference):
            messages: list[str] = []
            corrected = orient(flipped, reference, messages.append, "test image")
            self.assertLess(float(np.mean(np.abs(corrected - (reference ** 0.91)))), 1e-5)
            self.assertTrue(any("orientation corrected with vertical flip" in message for message in messages))

            ambiguous = np.full_like(reference, 0.08)
            preserved = orient(ambiguous, ambiguous, messages.append, "ambiguous image")
            self.assertTrue(np.array_equal(preserved, ambiguous))


if __name__ == "__main__":
    unittest.main()
