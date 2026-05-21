from pathlib import Path
from typing import List, Tuple, Union

import cv2

from config import CLASS_COLORS, DEFAULT_CONFIDENCE, MODEL_PATH


class DetectorError(RuntimeError):
    """Raised when model loading or image detection fails."""


class MaskDetector:
    def __init__(self, model_path: Path = MODEL_PATH, confidence: float = DEFAULT_CONFIDENCE):
        self.model_path = Path(model_path)
        self.confidence = confidence
        self.model = None

    def load_model(self) -> None:
        if self.model is not None:
            return

        if not self.model_path.exists():
            raise DetectorError(f"模型文件不存在：{self.model_path}")

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise DetectorError("未安装 ultralytics，请先执行 pip install -r requirements.txt") from exc

        self.model = YOLO(str(self.model_path))

    def detect_image(self, image_path: Union[str, Path]) -> Tuple[object, List[dict]]:
        self.load_model()

        image_path = Path(image_path)
        image = cv2.imread(str(image_path))
        if image is None:
            raise DetectorError(f"图片读取失败：{image_path}")

        try:
            predictions = self.model.predict(str(image_path), conf=self.confidence, verbose=False)
        except Exception as exc:
            raise DetectorError(f"模型推理失败：{exc}") from exc

        detections = self._parse_predictions(predictions)
        rendered_image = self._draw_detections(image, detections)
        return rendered_image, detections

    def _parse_predictions(self, predictions) -> List[dict]:
        detections = []
        if not predictions:
            return detections

        result = predictions[0]
        names = result.names

        for box in result.boxes:
            class_id = int(box.cls[0].item())
            class_name = names.get(class_id, str(class_id))
            confidence = float(box.conf[0].item())
            x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]

            detections.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidence,
                    "bbox": [x1, y1, x2, y2],
                }
            )

        return detections

    def _draw_detections(self, image, detections: List[dict]):
        for item in detections:
            x1, y1, x2, y2 = item["bbox"]
            class_name = item["class_name"]
            confidence = item["confidence"]
            color = CLASS_COLORS.get(class_name, (40, 120, 240))
            label = f"{class_name} {confidence:.2f}"

            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            label_y1 = max(y1 - label_size[1] - baseline, 0)
            cv2.rectangle(
                image,
                (x1, label_y1),
                (x1 + label_size[0], label_y1 + label_size[1] + baseline),
                color,
                -1,
            )
            cv2.putText(
                image,
                label,
                (x1, label_y1 + label_size[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

        return image
