from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

import cv2
import numpy as np
from PySide6.QtCore import QObject, QCameraPermission, QSize, QThread, QTimer, Qt, Signal, Slot
from PySide6.QtGui import QAction, QImage, QPixmap
from PySide6.QtMultimedia import QCamera, QCameraDevice, QMediaCaptureSession, QMediaDevices, QVideoSink
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)


MODEL_URL = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/object_detection_yolox/object_detection_yolox_2022nov.onnx"
MODEL_NAME = "object_detection_yolox_2022nov.onnx"
MIN_SCORE = 0.28
NMS_THRESHOLD = 0.45
OBJ_THRESHOLD = 0.4
INFERENCE_INTERVAL_MS = 80
MODEL_INPUT_SIZE = (640, 640)
DETECTION_STALE_SECONDS = 0.35

CLASS_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]


def app_support_dir() -> Path:
    return Path.home() / "Library" / "Application Support" / "mobilenet-object-detector"


@dataclass
class Detection:
    class_id: int
    label: str
    score: float
    x1: int
    y1: int
    x2: int
    y2: int


class VideoLabel(QLabel):
    def sizeHint(self) -> QSize:  # type: ignore[override]
        return QSize(960, 720)

    def minimumSizeHint(self) -> QSize:  # type: ignore[override]
        return QSize(640, 480)


class RealtimeDetector:
    def __init__(self) -> None:
        self._network: cv2.dnn.Net | None = None
        self._grids: np.ndarray | None = None
        self._expanded_strides: np.ndarray | None = None

    def _ensure_assets(self) -> Path:
        cache_dir = app_support_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)
        model_path = cache_dir / MODEL_NAME
        if model_path.exists() and model_path.stat().st_size < 1024 * 1024:
            model_path.unlink()
        if not model_path.exists():
            urlretrieve(MODEL_URL, model_path)
        return model_path

    def _ensure_model_loaded(self) -> None:
        if self._network is not None:
            return

        model_path = self._ensure_assets()
        self._network = cv2.dnn.readNet(str(model_path))
        self._generate_anchors()

    def _generate_anchors(self) -> None:
        grids: list[np.ndarray] = []
        expanded_strides: list[np.ndarray] = []
        for stride in (8, 16, 32):
            hsize = MODEL_INPUT_SIZE[1] // stride
            wsize = MODEL_INPUT_SIZE[0] // stride
            xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
            grid = np.stack((xv, yv), axis=2).reshape(1, -1, 2)
            grids.append(grid)
            expanded_strides.append(np.full((*grid.shape[:2], 1), stride, dtype=np.float32))
        self._grids = np.concatenate(grids, axis=1).astype(np.float32)
        self._expanded_strides = np.concatenate(expanded_strides, axis=1).astype(np.float32)

    def _preprocess(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, float]:
        frame_height, frame_width = frame_bgr.shape[:2]
        scale = min(MODEL_INPUT_SIZE[0] / frame_width, MODEL_INPUT_SIZE[1] / frame_height)
        resized_width = int(frame_width * scale)
        resized_height = int(frame_height * scale)
        resized = cv2.resize(frame_bgr, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

        padded = np.full((MODEL_INPUT_SIZE[1], MODEL_INPUT_SIZE[0], 3), 114, dtype=np.float32)
        padded[:resized_height, :resized_width] = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
        blob = np.transpose(padded, (2, 0, 1))[None, :, :, :]
        return blob, scale

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        self._ensure_model_loaded()
        if self._network is None or self._grids is None or self._expanded_strides is None:
            return []

        frame_height, frame_width = frame_bgr.shape[:2]
        blob, scale = self._preprocess(frame_bgr)
        self._network.setInput(blob)
        output = self._network.forward(self._network.getUnconnectedOutLayersNames())[0][0]
        output[:, :2] = (output[:, :2] + self._grids[0]) * self._expanded_strides[0]
        output[:, 2:4] = np.exp(output[:, 2:4]) * self._expanded_strides[0]

        boxes_xywh = np.ones_like(output[:, :4])
        boxes_xywh[:, 0] = output[:, 0] - output[:, 2] / 2
        boxes_xywh[:, 1] = output[:, 1] - output[:, 3] / 2
        boxes_xywh[:, 2] = output[:, 2]
        boxes_xywh[:, 3] = output[:, 3]

        score_matrix = output[:, 4:5] * output[:, 5:]
        max_scores = np.max(score_matrix, axis=1)
        class_ids_np = np.argmax(score_matrix, axis=1)

        boxes: list[list[int]] = []
        scores: list[float] = []
        class_ids: list[int] = []
        for box_xywh, score, class_id, obj_score in zip(boxes_xywh, max_scores, class_ids_np, output[:, 4]):
            if float(score) < MIN_SCORE or float(obj_score) < OBJ_THRESHOLD:
                continue
            x1 = int(box_xywh[0] / scale)
            y1 = int(box_xywh[1] / scale)
            box_width = int(box_xywh[2] / scale)
            box_height = int(box_xywh[3] / scale)
            x1 = max(0, min(frame_width - 1, x1))
            y1 = max(0, min(frame_height - 1, y1))
            box_width = max(1, min(frame_width - x1, box_width))
            box_height = max(1, min(frame_height - y1, box_height))
            boxes.append([x1, y1, box_width, box_height])
            scores.append(float(score))
            class_ids.append(int(class_id))

        if not boxes:
            return []

        kept = cv2.dnn.NMSBoxes(boxes, scores, MIN_SCORE, NMS_THRESHOLD)
        detections: list[Detection] = []
        for index in np.array(kept).flatten():
            x1, y1, width, height = boxes[int(index)]
            x2 = max(0, min(frame_width - 1, x1 + width))
            y2 = max(0, min(frame_height - 1, y1 + height))
            class_id = class_ids[int(index)]
            detections.append(
                Detection(
                    class_id=class_id,
                    label=CLASS_NAMES[class_id],
                    score=scores[int(index)],
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                )
            )

        detections.sort(key=lambda item: item.score, reverse=True)
        return detections


class DetectionWorker(QObject):
    detections_ready = Signal(object)
    status_changed = Signal(str)
    failed = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self._detector = RealtimeDetector()
        self._busy = False

    @Slot(object)
    def process_frame(self, payload: tuple[int, np.ndarray]) -> None:
        if self._busy:
            return

        frame_id, frame_bgr = payload
        self._busy = True
        try:
            status = "Loading YOLOX model..." if self._detector._network is None else "Detecting..."
            self.status_changed.emit(status)
            detections = self._detector.detect(frame_bgr)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))
        else:
            self.detections_ready.emit((frame_id, detections, time.monotonic()))
            self.status_changed.emit("Realtime detection active.")
        finally:
            self._busy = False


class MainWindow(QMainWindow):
    frame_available = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("MobileNet Object Detector")
        self.resize(1280, 820)
        self.setMinimumSize(980, 680)
        self.setStatusBar(QStatusBar())

        self._capture_session = QMediaCaptureSession(self)
        self._video_sink = QVideoSink(self)
        self._capture_session.setVideoSink(self._video_sink)
        self._camera: QCamera | None = None
        self._latest_frame_bgr: np.ndarray | None = None
        self._latest_detections: list[Detection] = []
        self._latest_frame_id = 0
        self._last_detection_frame_id = -1
        self._last_detection_time = 0.0
        self._inference_in_flight = False
        self._inference_enabled = False

        self._inference_timer = QTimer(self)
        self._inference_timer.timeout.connect(self._queue_inference)
        self._inference_timer.setInterval(INFERENCE_INTERVAL_MS)

        self._detection_thread = QThread(self)
        self._worker = DetectionWorker()
        self._worker.moveToThread(self._detection_thread)
        self.frame_available.connect(self._worker.process_frame)
        self._worker.detections_ready.connect(self._update_detections)
        self._worker.status_changed.connect(self.statusBar().showMessage)
        self._worker.failed.connect(self._handle_detection_failure)
        self._detection_thread.start()

        self._video_label = VideoLabel("Waiting for camera frames...")
        self._video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._video_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self._video_label.setStyleSheet("border: 1px solid #666; background: #000; color: #ddd;")

        self._camera_selector = QComboBox()
        self._camera_selector.currentIndexChanged.connect(self._switch_camera)

        self._refresh_button = QPushButton("Refresh Cameras")
        self._refresh_button.clicked.connect(self._reload_cameras)

        self._results = QPlainTextEdit()
        self._results.setReadOnly(True)
        self._results.setPlaceholderText("Live detections will appear here.")

        controls_layout = QFormLayout()
        controls_layout.addRow("Camera", self._camera_selector)

        actions_layout = QHBoxLayout()
        actions_layout.addWidget(self._refresh_button)

        right_layout = QVBoxLayout()
        right_layout.addLayout(controls_layout)
        right_layout.addLayout(actions_layout)
        right_layout.addWidget(QLabel("Live Detection Summary"))
        right_layout.addWidget(self._results)

        central_layout = QHBoxLayout()
        central_layout.addWidget(self._video_label, stretch=4)

        right_panel = QWidget()
        right_panel.setLayout(right_layout)
        right_panel.setMaximumWidth(360)
        right_panel.setMinimumWidth(300)
        central_layout.addWidget(right_panel, stretch=1)

        central = QWidget()
        central.setLayout(central_layout)
        self.setCentralWidget(central)

        reload_action = QAction("Reload Cameras", self)
        reload_action.triggered.connect(self._reload_cameras)
        self.menuBar().addAction(reload_action)

        self._video_sink.videoFrameChanged.connect(self._handle_video_frame)
        self._reload_cameras()

    @Slot()
    def _reload_cameras(self) -> None:
        devices = QMediaDevices.videoInputs()
        previous_id = self._camera_selector.currentData()

        self._camera_selector.blockSignals(True)
        self._camera_selector.clear()
        for device in devices:
            self._camera_selector.addItem(device.description(), device.id())
        self._camera_selector.blockSignals(False)

        if not devices:
            self._stop_camera()
            self._video_label.setText("No usable camera found.")
            self._results.setPlainText("No usable camera found.")
            self.statusBar().showMessage("No usable camera found.")
            return

        selected_index = 0
        if previous_id is not None:
            for index, device in enumerate(devices):
                if device.id() == previous_id:
                    selected_index = index
                    break

        self._camera_selector.setCurrentIndex(selected_index)
        self._start_camera(devices[selected_index])

    @Slot(int)
    def _switch_camera(self, combo_index: int) -> None:
        devices = QMediaDevices.videoInputs()
        if combo_index < 0 or combo_index >= len(devices):
            return
        self._start_camera(devices[combo_index])

    def _start_camera(self, device: QCameraDevice) -> None:
        permission = QCameraPermission()
        app = QApplication.instance()
        if app is None:
            return

        permission_status = app.checkPermission(permission)
        if permission_status == Qt.PermissionStatus.Undetermined:
            self.statusBar().showMessage("Requesting camera access from macOS...")
            app.requestPermission(permission, self, self._reload_cameras)
            return

        if permission_status != Qt.PermissionStatus.Granted:
            self._stop_camera()
            message = "Camera access is blocked. Enable it in System Settings > Privacy & Security > Camera."
            self._video_label.setText(message)
            self._results.setPlainText(message)
            self.statusBar().showMessage("Camera permission denied.")
            return

        self._stop_camera()
        self._camera = QCamera(device)
        self._camera.errorOccurred.connect(self._handle_camera_error)
        self._capture_session.setCamera(self._camera)
        self._camera.start()
        self._inference_enabled = True
        self._inference_timer.start()
        self.statusBar().showMessage(f"Using camera: {device.description()}")

    def _stop_camera(self) -> None:
        self._inference_timer.stop()
        self._inference_enabled = False
        if self._camera is None:
            return
        self._camera.stop()
        self._camera.deleteLater()
        self._camera = None
        self._capture_session.setCamera(None)

    @Slot()
    def _handle_camera_error(self) -> None:
        if self._camera is None:
            return
        message = self._camera.errorString() or "Unknown camera error."
        self._video_label.setText(message)
        self._results.setPlainText(message)
        self.statusBar().showMessage(message)

    @Slot()
    def _queue_inference(self) -> None:
        if not self._inference_enabled or self._latest_frame_bgr is None:
            return
        if self._inference_in_flight:
            return
        self._inference_in_flight = True
        self.frame_available.emit((self._latest_frame_id, self._latest_frame_bgr.copy()))

    @Slot(object)
    def _update_detections(self, payload: tuple[int, list[Detection], float]) -> None:
        frame_id, detections, detection_time = payload
        self._inference_in_flight = False
        self._last_detection_frame_id = frame_id
        self._last_detection_time = detection_time
        self._latest_detections = detections
        if detections:
            lines = [f"{item.label}: {item.score:.1%}" for item in detections[:10]]
            self._results.setPlainText("\n".join(lines))
            self._show_frame(self._latest_frame_bgr, detections)
        else:
            self._results.setPlainText("No objects detected above threshold.")
            self._show_frame(self._latest_frame_bgr, [])

        if self._latest_frame_bgr is not None and self._latest_frame_id != frame_id:
            self._queue_inference()

    @Slot(str)
    def _handle_detection_failure(self, message: str) -> None:
        self._results.setPlainText(message)
        self.statusBar().showMessage(f"Detection failed: {message}")

    @Slot(object)
    def _handle_video_frame(self, frame) -> None:  # type: ignore[no-untyped-def]
        if not frame.isValid():
            return

        image = frame.toImage()
        if image.isNull():
            return

        frame_bgr = self._qimage_to_bgr(image)
        self._latest_frame_bgr = frame_bgr
        self._latest_frame_id += 1

        detections = self._latest_detections
        if time.monotonic() - self._last_detection_time > DETECTION_STALE_SECONDS:
            detections = []
        self._show_frame(frame_bgr, detections)

    def _qimage_to_bgr(self, image: QImage) -> np.ndarray:
        rgb_image = image.convertToFormat(QImage.Format.Format_RGB888)
        width = rgb_image.width()
        height = rgb_image.height()
        bytes_per_line = rgb_image.bytesPerLine()
        ptr = rgb_image.bits()
        rgb = np.frombuffer(ptr, dtype=np.uint8).reshape((height, bytes_per_line))
        rgb = rgb[:, : width * 3].reshape((height, width, 3))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def _annotate_frame(self, frame_bgr: np.ndarray, detections: list[Detection]) -> np.ndarray:
        annotated = frame_bgr.copy()
        for detection in detections:
            cv2.rectangle(annotated, (detection.x1, detection.y1), (detection.x2, detection.y2), (0, 220, 0), 2)
            label = f"{detection.label} {detection.score:.0%}"
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            text_y = detection.y1 - 10 if detection.y1 - 10 > text_height else detection.y1 + text_height + 10
            cv2.rectangle(
                annotated,
                (detection.x1, text_y - text_height - baseline - 4),
                (detection.x1 + text_width + 8, text_y + baseline - 4),
                (0, 220, 0),
                thickness=-1,
            )
            cv2.putText(
                annotated,
                label,
                (detection.x1 + 4, text_y - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )
        return annotated

    def _show_frame(self, frame_bgr: np.ndarray | None, detections: list[Detection]) -> None:
        if frame_bgr is None:
            return
        annotated = self._annotate_frame(frame_bgr, detections)
        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        image = QImage(rgb.data, width, height, channels * width, QImage.Format.Format_RGB888)
        scaled = image.scaled(self._video_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self._video_label.setPixmap(QPixmap.fromImage(scaled))

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._stop_camera()
        self._detection_thread.quit()
        self._detection_thread.wait(3000)
        super().closeEvent(event)


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
