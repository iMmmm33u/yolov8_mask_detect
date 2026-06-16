from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "mask_yolo.pt"
DATASET_CONFIG_PATH = PROJECT_ROOT / "datasets" / "mask_dataset" / "mask.yaml"

APP_TITLE = "口罩检测系统"
SUPPORTED_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic", ".heif")
DEFAULT_CONFIDENCE = 0.25

CLASS_COLORS = {
    "mask": (0, 180, 0),
    "no_mask": (220, 0, 0),
}
