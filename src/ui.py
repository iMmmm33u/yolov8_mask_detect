from pathlib import Path
from typing import Optional

import cv2
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from config import APP_TITLE, SUPPORTED_IMAGE_EXTENSIONS
from detector import DetectorError, MaskDetector
from utils import cv_image_to_pixmap, format_detection_results, is_supported_image


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.detector = MaskDetector()
        self.current_image_path: Optional[Path] = None

        self.setWindowTitle(APP_TITLE)
        self.resize(980, 720)
        self._build_ui()

    def _build_ui(self) -> None:
        root = QWidget()
        main_layout = QVBoxLayout(root)

        title_label = QLabel(APP_TITLE)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setObjectName("titleLabel")

        self.image_label = QLabel("请选择一张图片")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(760, 440)
        self.image_label.setObjectName("imageLabel")

        button_layout = QHBoxLayout()
        self.select_button = QPushButton("选择图片")
        self.detect_button = QPushButton("开始检测")
        self.clear_button = QPushButton("清空结果")
        self.exit_button = QPushButton("退出系统")

        button_layout.addWidget(self.select_button)
        button_layout.addWidget(self.detect_button)
        button_layout.addWidget(self.clear_button)
        button_layout.addWidget(self.exit_button)

        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setPlaceholderText("检测结果将在这里显示")
        self.result_text.setMinimumHeight(120)

        main_layout.addWidget(title_label)
        main_layout.addWidget(self.image_label, stretch=1)
        main_layout.addLayout(button_layout)
        main_layout.addWidget(self.result_text)

        self.setCentralWidget(root)
        self._connect_signals()
        self._set_styles()

    def _connect_signals(self) -> None:
        self.select_button.clicked.connect(self.select_image)
        self.detect_button.clicked.connect(self.detect_image)
        self.clear_button.clicked.connect(self.clear_result)
        self.exit_button.clicked.connect(self.close)

    def _set_styles(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow {
                background: #f5f7fa;
            }
            QLabel#titleLabel {
                font-size: 24px;
                font-weight: 700;
                padding: 12px;
                color: #1f2937;
            }
            QLabel#imageLabel {
                background: #ffffff;
                border: 1px solid #d1d5db;
                color: #6b7280;
                font-size: 16px;
            }
            QPushButton {
                min-height: 36px;
                padding: 0 18px;
                font-size: 14px;
                background: #2563eb;
                color: white;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover {
                background: #1d4ed8;
            }
            QTextEdit {
                background: #ffffff;
                border: 1px solid #d1d5db;
                font-size: 14px;
                padding: 8px;
            }
            """
        )

    def select_image(self) -> None:
        file_filter = "图片文件 (*.jpg *.jpeg *.png *.bmp *.webp)"
        image_path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", file_filter)
        if not image_path:
            return

        if not is_supported_image(image_path, SUPPORTED_IMAGE_EXTENSIONS):
            QMessageBox.warning(self, "文件格式错误", "请选择 jpg、jpeg、png、bmp 或 webp 图片。")
            return

        self.current_image_path = Path(image_path)
        image = cv2.imread(str(self.current_image_path))
        if image is None:
            QMessageBox.warning(self, "图片读取失败", "无法读取所选图片，请更换图片。")
            return

        self._show_image(image)
        self.result_text.setText(f"已选择图片：{self.current_image_path}")

    def detect_image(self) -> None:
        if self.current_image_path is None:
            QMessageBox.information(self, "提示", "请先选择一张图片。")
            return

        try:
            result_image, detections = self.detector.detect_image(self.current_image_path)
        except DetectorError as exc:
            QMessageBox.warning(self, "检测失败", str(exc))
            return

        self._show_image(result_image)
        self.result_text.setText(format_detection_results(detections))

    def clear_result(self) -> None:
        self.current_image_path = None
        self.image_label.clear()
        self.image_label.setText("请选择一张图片")
        self.result_text.clear()

    def _show_image(self, image) -> None:
        pixmap = cv_image_to_pixmap(
            image,
            self.image_label.width(),
            self.image_label.height(),
        )
        self.image_label.setPixmap(pixmap)
