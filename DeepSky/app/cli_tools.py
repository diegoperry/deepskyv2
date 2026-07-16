from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Callable


LogCallback = Callable[[str], None]

# Keep command construction here so CLI argument formats can be adjusted after testing.
DEEPSNR_COMMAND = ["{exe}", "-i", "{input}", "-o", "{output}"]
STARNET_COMMAND = ["{exe}", "-i", "{input}", "-o", "{output}"]
REALESRGAN_COMMAND = ["{exe}", "-i", "{input}", "-o", "{output}", "-n", "{model}"]


class ToolExecutionError(RuntimeError):
    def __init__(self, tool_name: str, return_code: int, output: str) -> None:
        self.tool_name = tool_name
        self.return_code = return_code
        self.output = output
        detail = f"{tool_name} exited with code {return_code}"
        if output.strip():
            detail = f"{detail}: {output.strip()[-800:]}"
        super().__init__(detail)


def find_executable(tool_folder: Path) -> Path | None:
    tool_folder = Path(tool_folder)
    if tool_folder.is_file() and tool_folder.suffix.lower() == ".exe":
        return tool_folder
    if not tool_folder.exists():
        return None

    exes = sorted(tool_folder.rglob("*.exe"))
    if not exes:
        return None

    preferred = [
        exe for exe in exes
        if "starnet" in exe.name.lower()
        or "deepsnr" in exe.name.lower()
        or "realesrgan" in exe.name.lower()
    ]
    return preferred[0] if preferred else exes[0]


def get_help_output(executable_path: Path) -> str:
    exe = Path(executable_path)
    if not exe.exists():
        return "Executable not found."

    attempts = ([str(exe), "--help"], [str(exe), "-h"], [str(exe)])
    output_parts: list[str] = []
    for command in attempts:
        try:
            completed = subprocess.run(
                command,
                cwd=str(exe.parent),
                capture_output=True,
                text=True,
                timeout=8,
                check=False,
            )
        except Exception as exc:
            output_parts.append(f"{' '.join(command)} failed: {exc}")
            continue

        combined = (completed.stdout or "") + (completed.stderr or "")
        if combined.strip():
            return combined.strip()
        output_parts.append(f"{' '.join(command)} exited {completed.returncode} with no output.")
    return "\n".join(output_parts)


def _format_command(template: list[str], exe: Path, input_path: Path, output_path: Path) -> list[str]:
    values = {
        "exe": str(exe),
        "input": str(input_path),
        "output": str(output_path),
        "model": "realesrgan-x4plus",
    }
    return [part.format(**values) for part in template]


def _run_tool(
    template: list[str],
    executable_path: Path,
    input_path: Path,
    output_path: Path,
    log: LogCallback | None = None,
) -> None:
    exe = Path(executable_path)
    command = _format_command(template, exe, input_path, output_path)
    if log:
        log(f"Running: {' '.join(command)}")

    process = subprocess.Popen(
        command,
        cwd=str(exe.parent),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert process.stdout is not None
    output_lines: list[str] = []
    for line in process.stdout:
        output_lines.append(line.rstrip())
        if log:
            log(line.rstrip())
    return_code = process.wait()
    if return_code != 0:
        raise ToolExecutionError(exe.name, return_code, "\n".join(output_lines))
    if not output_path.exists():
        raise RuntimeError(f"{exe.name} finished but did not create {output_path.name}")


def run_deepsnr(
    input_path: Path,
    output_path: Path,
    executable_path: Path,
    log: LogCallback | None = None,
    *,
    model: int | None = None,
    stride: int | None = None,
) -> None:
    command = list(DEEPSNR_COMMAND)
    if model is not None:
        if model not in {1, 2}:
            raise ValueError("DeepSNR model must be 1 or 2.")
        command.extend(["--model", str(model)])
    if stride is not None:
        if stride < 2 or stride > 512 or stride % 2:
            raise ValueError("DeepSNR stride must be an even integer between 2 and 512.")
        command.extend(["--stride", str(stride)])
    _run_tool(command, executable_path, input_path, output_path, log)


def run_starnet(
    input_path: Path,
    output_path: Path,
    executable_path: Path,
    log: LogCallback | None = None,
) -> None:
    _run_tool(STARNET_COMMAND, executable_path, input_path, output_path, log)


def run_realesrgan(
    input_path: Path,
    output_path: Path,
    executable_path: Path,
    log: LogCallback | None = None,
    *,
    model: str = "realesrgan-x4plus",
) -> None:
    if model not in {"realesrgan-x4plus", "realesrnet-x4plus", "realesrgan-x4plus-anime"}:
        raise ValueError("Unsupported Real-ESRGAN model.")
    command = [part if part != "{model}" else model for part in REALESRGAN_COMMAND]
    _run_tool(command, executable_path, input_path, output_path, log)
