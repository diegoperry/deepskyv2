from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from app.cli_tools import run_deepsnr, run_realesrgan


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

    @patch("app.cli_tools._run_tool")
    def test_realesrgan_model_is_forwarded_to_cli(self, run_tool) -> None:
        executable = Path("realesrgan-ncnn-vulkan.exe")
        source = Path("final.png")
        output = Path("pixel_restored.png")

        run_realesrgan(source, output, executable, model="realesrgan-x4plus")

        command, exe_arg, source_arg, output_arg, log_arg = run_tool.call_args.args
        self.assertEqual(command, ["{exe}", "-i", "{input}", "-o", "{output}", "-n", "realesrgan-x4plus"])
        self.assertEqual((exe_arg, source_arg, output_arg), (executable, source, output))
        self.assertIsNone(log_arg)

    def test_invalid_realesrgan_model_is_rejected_before_launch(self) -> None:
        with self.assertRaises(ValueError):
            run_realesrgan(
                Path("final.png"),
                Path("pixel_restored.png"),
                Path("realesrgan-ncnn-vulkan.exe"),
                model="paint-the-nebula",
            )


if __name__ == "__main__":
    unittest.main()
