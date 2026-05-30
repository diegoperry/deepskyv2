from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot

from .pipeline import PipelineMode, run_pipeline
from .settings import AppSettings


class PipelineWorker(QObject):
    log = Signal(str)
    finished = Signal(dict)
    failed = Signal(str)

    def __init__(self, input_path: Path, settings: AppSettings, mode: PipelineMode) -> None:
        super().__init__()
        self.input_path = input_path
        self.settings = settings
        self.mode = mode

    @Slot()
    def run(self) -> None:
        try:
            result = run_pipeline(self.input_path, self.settings, self.mode, self.log.emit)
        except Exception as exc:
            self.failed.emit(str(exc))
        else:
            self.finished.emit({key: str(value) for key, value in result.items()})

