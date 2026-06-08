from __future__ import annotations

import subprocess
from pathlib import Path

from PySide6.QtCore import Qt, QThread
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QFrame,
    QSizePolicy,
    QSlider,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .cli_tools import find_executable
from .image_io import SUPPORTED_INPUTS, make_preview
from .pipeline import PipelineMode
from .settings import AppSettings, load_settings, save_settings
from .siril_cli import find_siril_executable
from .worker import PipelineWorker


class PreviewLabel(QLabel):
    def __init__(self, title: str) -> None:
        super().__init__(title)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(360, 260)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("border: 1px solid #313845; background: #11151b; color: #b8c0cc;")
        self._pixmap: QPixmap | None = None

    def set_image(self, path: Path) -> None:
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            self.setText("Preview unavailable")
            self._pixmap = None
            return
        self._pixmap = pixmap
        self._rescale()

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._rescale()

    def _rescale(self) -> None:
        if self._pixmap:
            scaled = self._pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.setPixmap(scaled)


class DropZone(QFrame):
    def __init__(self, pick_callback, drop_callback) -> None:
        super().__init__()
        self.pick_callback = pick_callback
        self.drop_callback = drop_callback
        self.setAcceptDrops(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumHeight(150)
        self.setObjectName("dropZone")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(8)

        title = QLabel("Drag & drop your astrophotography file")
        title.setObjectName("dropTitle")
        title.setAlignment(Qt.AlignCenter)
        title.setWordWrap(True)
        detail = QLabel("Supports FITS and TIFF formats")
        detail.setObjectName("dropDetail")
        detail.setAlignment(Qt.AlignCenter)

        layout.addStretch(1)
        layout.addWidget(title)
        layout.addWidget(detail)
        layout.addStretch(1)

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.LeftButton:
            self.pick_callback()
        super().mousePressEvent(event)

    def dragEnterEvent(self, event) -> None:  # type: ignore[override]
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event) -> None:  # type: ignore[override]
        for url in event.mimeData().urls():
            path = Path(url.toLocalFile())
            if path.suffix.lower() in SUPPORTED_INPUTS:
                self.drop_callback(path)
                event.acceptProposedAction()
                return
        event.ignore()


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("DeepSky")
        self.resize(1180, 780)
        self.settings = load_settings()
        self.input_path: Path | None = None
        self.last_output_folder: Path | None = None
        self.thread: QThread | None = None
        self.worker: PipelineWorker | None = None

        self.input_edit = QLineEdit()
        self.output_edit = QLineEdit(self.settings.output_folder)
        self.starnet_edit = QLineEdit(self.settings.starnet_folder)
        self.deepsnr_edit = QLineEdit(self.settings.deepsnr_folder)
        self.siril_edit = QLineEdit(self.settings.siril_folder)
        self.color_mode_combo = QComboBox()
        self.color_mode_combo.addItems(["Off", "Basic", "Siril Photometric"])
        self.color_mode_combo.setCurrentText(self.settings.color_calibration_mode)
        self.object_name_edit = QLineEdit(self.settings.siril_object_name)
        self.ra_dec_edit = QLineEdit(self.settings.siril_ra_dec)
        self.focal_length_edit = QLineEdit(self.settings.siril_focal_length)
        self.pixel_size_edit = QLineEdit(self.settings.siril_pixel_size)
        self.apply_scnr_check = QCheckBox("Apply SCNR / Remove Green")
        self.apply_scnr_check.setChecked(self.settings.siril_apply_scnr)
        self.siril_deconvolution_check = QCheckBox("Test Siril Galaxy Deconvolution")
        self.siril_deconvolution_check.setChecked(self.settings.siril_deconvolution_enabled)
        self.debug_mode_check = QCheckBox("Debug mode")
        self.debug_mode_check.setChecked(self.settings.siril_debug_mode)
        self.saturation_slider = QSlider(Qt.Horizontal)
        self.saturation_slider.setRange(0, 100)
        self.saturation_slider.setValue(self.settings.siril_color_saturation)
        self.saturation_value = QLabel(str(self.settings.siril_color_saturation))
        self.background_smoothness_slider = QSlider(Qt.Horizontal)
        self.background_smoothness_slider.setRange(0, 100)
        self.background_smoothness_slider.setValue(self.settings.galaxy_background_smoothness)
        self.background_smoothness_value = QLabel(str(self.settings.galaxy_background_smoothness))
        self.background_darkness_slider = QSlider(Qt.Horizontal)
        self.background_darkness_slider.setRange(0, 100)
        self.background_darkness_slider.setValue(self.settings.galaxy_background_darkness)
        self.background_darkness_value = QLabel(str(self.settings.galaxy_background_darkness))
        self.chroma_noise_slider = QSlider(Qt.Horizontal)
        self.chroma_noise_slider.setRange(0, 100)
        self.chroma_noise_slider.setValue(self.settings.galaxy_chroma_noise_reduction)
        self.chroma_noise_value = QLabel(str(self.settings.galaxy_chroma_noise_reduction))
        self.protect_galaxy_check = QCheckBox("Protect Galaxy Detail")
        self.protect_galaxy_check.setChecked(self.settings.galaxy_protect_detail)
        self.status_label = QLabel()
        self.before_preview = PreviewLabel("Before preview")
        self.after_preview = PreviewLabel("After preview")
        self.drop_zone = DropZone(self.pick_input_file, self.set_input_file)
        self.log_panel = QTextEdit()
        self.log_panel.setReadOnly(True)

        self.run_buttons: list[QPushButton] = []
        self._build_ui()
        self._apply_style()
        self.check_tools()

    def _build_ui(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(26, 24, 26, 22)
        root.setSpacing(16)

        hero = QWidget()
        hero_layout = QVBoxLayout(hero)
        hero_layout.setContentsMargins(0, 0, 0, 0)
        hero_layout.setSpacing(12)
        hero_layout.setAlignment(Qt.AlignCenter)

        pill = QLabel("●  DeepSky Astrophotography Pipeline v1.0")
        pill.setObjectName("heroPill")
        pill.setAlignment(Qt.AlignCenter)
        title = QLabel('Process the <span style="color:#7fa2ff;">Cos</span><span style="color:#b892ff;">mo</span><span style="color:#ff826d;">s</span>')
        title.setObjectName("heroTitle")
        title.setAlignment(Qt.AlignCenter)
        subtitle = QLabel("Automated processing for deep-sky imagers. Drop your FITS or TIFF files and reveal cleaner stars, galaxies, and nebulae.")
        subtitle.setObjectName("heroSubtitle")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setWordWrap(True)

        hero_layout.addWidget(pill, 0, Qt.AlignCenter)
        hero_layout.addWidget(title)
        hero_layout.addWidget(subtitle)
        root.addWidget(hero)

        top_row = QHBoxLayout()
        top_row.setSpacing(18)
        top_row.addStretch(1)
        top_row.addWidget(self.drop_zone, 2)
        top_row.addStretch(1)
        root.addLayout(top_row)

        actions = QHBoxLayout()
        actions.setSpacing(10)
        actions.addStretch(1)
        button = QPushButton("Run Full Pipeline")
        button.setObjectName("primaryCta")
        button.clicked.connect(lambda checked=False: self.start_pipeline(PipelineMode.FULL))
        self.run_buttons.append(button)
        actions.addWidget(button)
        actions.addStretch(1)
        root.addLayout(actions)

        previews = QHBoxLayout()
        previews.setSpacing(16)
        before_group = QGroupBox("Before")
        before_group.setObjectName("previewGroup")
        before_layout = QVBoxLayout(before_group)
        before_layout.addWidget(self.before_preview)
        after_group = QGroupBox("After")
        after_group.setObjectName("previewGroup")
        after_layout = QVBoxLayout(after_group)
        after_layout.addWidget(self.after_preview)
        previews.addWidget(before_group, 1)
        previews.addWidget(after_group, 1)
        root.addLayout(previews, 3)

        log_group = QGroupBox("Processing Log")
        log_layout = QVBoxLayout(log_group)
        log_layout.addWidget(self.log_panel)
        root.addWidget(log_group, 1)

        self.setCentralWidget(central)

    def _picker_row(self, edit: QLineEdit, callback) -> QWidget:
        wrapper = QWidget()
        layout = QHBoxLayout(wrapper)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(edit, 1)
        button = QPushButton("Browse")
        button.clicked.connect(callback)
        layout.addWidget(button)
        return wrapper

    def _slider_row(self, slider: QSlider, value_label: QLabel) -> QWidget:
        wrapper = QWidget()
        layout = QHBoxLayout(wrapper)
        layout.setContentsMargins(0, 0, 0, 0)
        value_label.setMinimumWidth(28)
        value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        slider.valueChanged.connect(lambda value, label=value_label: label.setText(str(value)))
        layout.addWidget(slider, 1)
        layout.addWidget(value_label)
        return wrapper

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background: #080d16; color: #edf1f7; font-size: 13px; }
            QLabel#heroPill {
                background: #101a32;
                border: 1px solid #234ea8;
                border-radius: 14px;
                padding: 6px 16px;
                color: #8db5ff;
                font-family: Consolas, monospace;
                font-size: 12px;
            }
            QLabel#heroTitle {
                color: #f8fbff;
                font-size: 54px;
                font-weight: 800;
                letter-spacing: 0px;
            }
            QLabel#heroSubtitle {
                color: #7f97bd;
                font-size: 17px;
                max-width: 780px;
            }
            QFrame#dropZone {
                background: #0d1420;
                border: 2px dashed #263957;
                border-radius: 14px;
                max-width: 720px;
            }
            QFrame#dropZone:hover {
                border-color: #4f86ff;
                background: #101a2a;
            }
            QLabel#dropTitle {
                color: #f8fbff;
                font-size: 26px;
                font-weight: 800;
            }
            QLabel#dropDetail {
                color: #7f97bd;
                font-family: Consolas, monospace;
                font-size: 13px;
            }
            QGroupBox { border: 1px solid #243147; border-radius: 8px; margin-top: 10px; padding: 12px; background: #0a101a; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; color: #d7dde8; font-weight: 700; }
            QGroupBox#previewGroup { background: #080d16; border-color: #26354c; }
            QTextEdit { min-height: 74px; max-height: 120px; }
            QLineEdit, QTextEdit { background: #070b12; border: 1px solid #26354c; border-radius: 6px; padding: 7px; color: #f3f6fb; }
            QComboBox { background: #070b12; border: 1px solid #26354c; border-radius: 6px; padding: 7px; color: #f3f6fb; }
            QCheckBox { spacing: 7px; }
            QSlider::groove:horizontal { background: #26354c; height: 5px; border-radius: 2px; }
            QSlider::handle:horizontal { background: #d7dde8; width: 16px; margin: -6px 0; border-radius: 8px; }
            QSlider::sub-page:horizontal { background: #4f86ff; border-radius: 2px; }
            QPushButton { background: #2d6cdf; border: 0; border-radius: 7px; padding: 9px 12px; color: white; font-weight: 700; }
            QPushButton#primaryCta { min-width: 280px; padding: 13px 20px; font-size: 15px; }
            QPushButton:hover { background: #4f86ff; }
            QPushButton:disabled { background: #46505f; color: #a8b0bb; }
            QLabel { color: #d8dee8; }
            """
        )

    def set_input_file(self, path: Path) -> None:
        self.input_path = path
        self.input_edit.setText(str(path))
        preview_path = Path(self.output_edit.text().strip() or self.settings.output_folder) / "_preview_before.png"
        try:
            make_preview(self.input_path, preview_path)
            self.before_preview.set_image(preview_path)
        except Exception as exc:
            self.append_log(f"Preview failed: {exc}")

    def pick_input_file(self) -> None:
        filters = "Astro images (*.fits *.fit *.fts *.tif *.tiff);;All files (*.*)"
        path, _ = QFileDialog.getOpenFileName(self, "Select input image", "", filters)
        if not path:
            return
        self.set_input_file(Path(path))

    def pick_output_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select output folder", self.output_edit.text())
        if path:
            self.output_edit.setText(path)
            self.save_current_settings()

    def pick_starnet_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select StarNet folder", self.starnet_edit.text())
        if path:
            self.starnet_edit.setText(path)
            self.save_current_settings()
            self.check_tools()

    def pick_deepsnr_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select DeepSNR folder", self.deepsnr_edit.text())
        if path:
            self.deepsnr_edit.setText(path)
            self.save_current_settings()
            self.check_tools()

    def pick_siril_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select Siril folder", self.siril_edit.text())
        if path:
            self.siril_edit.setText(path)
            self.save_current_settings()
            self.check_tools()

    def save_current_settings(self) -> AppSettings:
        self.settings = AppSettings(
            starnet_folder=self.starnet_edit.text().strip(),
            deepsnr_folder=self.deepsnr_edit.text().strip(),
            siril_folder=self.siril_edit.text().strip(),
            output_folder=self.output_edit.text().strip(),
            color_calibration_mode=self.color_mode_combo.currentText(),
            siril_object_name=self.object_name_edit.text().strip(),
            siril_ra_dec=self.ra_dec_edit.text().strip(),
            siril_focal_length=self.focal_length_edit.text().strip(),
            siril_pixel_size=self.pixel_size_edit.text().strip(),
            siril_apply_scnr=self.apply_scnr_check.isChecked(),
            siril_color_saturation=self.saturation_slider.value(),
            siril_deconvolution_enabled=self.siril_deconvolution_check.isChecked(),
            siril_debug_mode=self.debug_mode_check.isChecked(),
            galaxy_background_smoothness=self.background_smoothness_slider.value(),
            galaxy_background_darkness=self.background_darkness_slider.value(),
            galaxy_chroma_noise_reduction=self.chroma_noise_slider.value(),
            galaxy_protect_detail=self.protect_galaxy_check.isChecked(),
            input_processing_mode=getattr(self.settings, "input_processing_mode", "Auto"),
            stretch_level=getattr(self.settings, "stretch_level", "Standard"),
            telescope_profile=getattr(self.settings, "telescope_profile", "Auto"),
            prestretched_input=False,
            object_type=getattr(self.settings, "object_type", "Nebula"),
        )
        save_settings(self.settings)
        return self.settings

    def check_tools(self) -> None:
        settings = self.save_current_settings()
        starnet = find_executable(Path(settings.starnet_folder))
        deepsnr = find_executable(Path(settings.deepsnr_folder))
        siril = find_siril_executable(Path(settings.siril_folder))
        parts = [
            f"StarNet: {'OK - ' + starnet.name if starnet else 'missing'}",
            f"DeepSNR: {'OK - ' + deepsnr.name if deepsnr else 'missing'}",
            f"Siril: {'OK - ' + siril.name if siril else 'missing'}",
        ]
        self.status_label.setText(" | ".join(parts))

    def start_pipeline(self, mode: PipelineMode) -> None:
        input_text = self.input_edit.text().strip()
        if not input_text:
            QMessageBox.warning(self, "Input required", "Choose a FITS or TIFF file first.")
            return
        input_path = Path(input_text)
        if input_path.suffix.lower() not in SUPPORTED_INPUTS:
            QMessageBox.warning(self, "Unsupported file", "Choose a FITS or TIFF file.")
            return

        self.save_current_settings()
        self.set_running(True)
        self.log_panel.clear()
        self.append_log(f"Starting {mode.value} pipeline.")

        self.thread = QThread()
        self.worker = PipelineWorker(input_path, self.settings, mode)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.log.connect(self.append_log)
        self.worker.finished.connect(self.pipeline_finished)
        self.worker.failed.connect(self.pipeline_failed)
        self.worker.finished.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def pipeline_finished(self, result: dict) -> None:
        self.set_running(False)
        self.last_output_folder = Path(result["job_folder"])
        self.before_preview.set_image(Path(result["before_preview"]))
        self.after_preview.set_image(Path(result["after_preview"]))
        self.append_log(f"Output folder: {self.last_output_folder}")
        QMessageBox.information(self, "DeepSky", "Processing complete.")

    def pipeline_failed(self, message: str) -> None:
        self.set_running(False)
        self.append_log(f"ERROR: {message}")
        QMessageBox.critical(self, "Processing failed", message)

    def set_running(self, running: bool) -> None:
        for button in self.run_buttons:
            button.setDisabled(running)

    def append_log(self, message: str) -> None:
        self.log_panel.append(message)

    def open_output_folder(self) -> None:
        folder = self.last_output_folder or Path(self.output_edit.text().strip())
        folder.mkdir(parents=True, exist_ok=True)
        subprocess.Popen(["explorer", str(folder)])
