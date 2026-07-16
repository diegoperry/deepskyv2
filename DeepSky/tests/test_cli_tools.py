from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from app.cli_tools import run_deepsnr


class DeepSnrCommandTests(unittest.TestCase):
    @patch("app.cli_tools._run_tool")
    def test_model_and_stride_are_forwarded_to_cli(self, run_tool) -> None:
        executable = Path("deepsnr.exe")
        source = Path("linear_rgb.tif")
        output = Path("denoised.tif")

        run_deepsnr(source, output, executable, model=2, stride=256)

        command, exe_arg, source_arg, output_arg, log_arg = run_tool.call_args.args
        self.assertEqual(command[-4:], ["--model", "2", "--stride", "256"])
        self.assertEqual((exe_arg, source_arg, output_arg), (executable, source, output))
        self.assertIsNone(log_arg)

    def test_invalid_stride_is_rejected_before_launch(self) -> None:
        with self.assertRaises(ValueError):
            run_deepsnr(
                Path("input.tif"),
                Path("output.tif"),
                Path("deepsnr.exe"),
                model=2,
                stride=255,
            )


if __name__ == "__main__":
    unittest.main()
