from pathlib import Path
from typing import Iterable, List, Union

import cv2
from PyQt5.QtGui import QImage, QPixmap


def is_supported_image(path: Union[str, Path], extensions: Iterable[str]) -> bool:
    return Path(path).suffix.lower() in extensions


def cv_image_to_pixmap(image_bgr, max_width: int, max_height: int) -> QPixmap:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    height, width, channels = image_rgb.shape
    bytes_per_line = channels * width
    qimage = QImage(
        image_rgb.data,
        width,
        height,
        bytes_per_line,
        QImage.Format_RGB888,
    )
    pixmap = QPixmap.fromImage(qimage.copy())
    return pixmap.scaled(max_width, max_height, aspectRatioMode=1, transformMode=1)


def summarize_mask_status(results: List[dict]) -> str:
    if not results:
        return "佩戴情况：未检测到人脸或口罩相关目标，无法判断。"

    mask_count = sum(1 for item in results if item["class_name"] == "mask")
    no_mask_count = sum(1 for item in results if item["class_name"] == "no_mask")
    unknown_count = len(results) - mask_count - no_mask_count

    if no_mask_count == 0 and mask_count > 0:
        status = "全部已佩戴口罩"
    elif mask_count == 0 and no_mask_count > 0:
        status = "存在未佩戴口罩人员"
    elif mask_count > 0 and no_mask_count > 0:
        status = "部分人员未佩戴口罩"
    else:
        status = "检测到未知类别，无法判断"

    details = f"已佩戴 {mask_count} 人，未佩戴 {no_mask_count} 人"
    if unknown_count:
        details += f"，未知 {unknown_count} 个"

    return f"佩戴情况：{status}（{details}）。"


def format_detection_results(results: List[dict]) -> str:
    if not results:
        return summarize_mask_status(results)

    lines = [
        summarize_mask_status(results),
        f"检测完成，共检测到 {len(results)} 个目标。",
    ]
    for index, item in enumerate(results, start=1):
        lines.append(
            f"{index}. 类别：{item['class_name']}，"
            f"置信度：{item['confidence']:.2f}，"
            f"位置：{item['bbox']}"
        )
    return "\n".join(lines)
