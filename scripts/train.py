import argparse
import os
import platform
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO mask detection model.")
    parser.add_argument("--model", default="yolov8n.pt", help="Initial YOLO model or checkpoint.")
    parser.add_argument("--data", default="datasets/mask_dataset/mask.yaml", help="YOLO dataset yaml path.")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument("--device", default="0", help="Training device, such as 0, 0,1, cpu.")
    parser.add_argument("--workers", type=int, default=8, help="Data loading workers.")
    parser.add_argument("--project", default="runs/detect", help="Output project directory.")
    parser.add_argument("--name", default="mask_train", help="Output experiment name.")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    parser.add_argument("--skip-data-check", action="store_true", help="Skip dataset image/label matching check.")
    parser.add_argument("--verbose", action="store_true", help="Show verbose Ultralytics training logs.")
    return parser.parse_args()


def require_tqdm():
    try:
        from tqdm.auto import tqdm
    except ImportError as exc:
        raise RuntimeError("未安装 tqdm，请先执行 pip install tqdm 或重新安装项目依赖。") from exc

    return tqdm


def iter_images(directory: Path) -> Iterable[Path]:
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return (path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in image_extensions)


def check_dataset(data_path: Path, tqdm) -> None:
    dataset_root = data_path.parent
    split_dirs = {
        "train": ("images/train", "labels/train"),
        "val": ("images/val", "labels/val"),
        "test": ("images/test", "labels/test"),
    }

    for split_name, (image_subdir, label_subdir) in tqdm(split_dirs.items(), desc="检查数据集", unit="split"):
        image_dir = dataset_root / image_subdir
        label_dir = dataset_root / label_subdir
        if not image_dir.exists():
            raise FileNotFoundError(f"{split_name} 图片目录不存在：{image_dir}")
        if not label_dir.exists():
            raise FileNotFoundError(f"{split_name} 标注目录不存在：{label_dir}")

        images = list(iter_images(image_dir))
        missing_labels = [image.name for image in images if not (label_dir / f"{image.stem}.txt").exists()]
        if missing_labels:
            examples = ", ".join(missing_labels[:5])
            raise FileNotFoundError(f"{split_name} 有 {len(missing_labels)} 张图片缺少标注文件，例如：{examples}")

        label_count = len(list(label_dir.glob("*.txt")))
        tqdm.write(f"{split_name}: images={len(images)} labels={label_count}")


def add_training_progress(model, epochs: int, tqdm) -> None:
    progress = {"bar": None}

    def on_train_start(_trainer) -> None:
        progress["bar"] = tqdm(total=epochs, desc="训练进度", unit="epoch")

    def on_train_epoch_end(_trainer) -> None:
        bar = progress["bar"]
        if bar is not None:
            bar.update(1)

    def on_train_end(_trainer) -> None:
        bar = progress["bar"]
        if bar is not None:
            bar.close()

    model.add_callback("on_train_start", on_train_start)
    model.add_callback("on_train_epoch_end", on_train_epoch_end)
    model.add_callback("on_train_end", on_train_end)


def main() -> int:
    if platform.system() == "Darwin":
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    args = parse_args()
    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path

    project_path = Path(args.project)
    if not project_path.is_absolute():
        project_path = PROJECT_ROOT / project_path

    if not data_path.exists():
        raise FileNotFoundError(f"数据集配置文件不存在：{data_path}")

    tqdm = None
    if not args.no_progress:
        tqdm = require_tqdm()
        if not args.skip_data_check:
            check_dataset(data_path, tqdm)

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("未安装 ultralytics，请先安装项目依赖。") from exc

    model = YOLO(args.model)
    if tqdm is not None:
        add_training_progress(model, args.epochs, tqdm)

    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(project_path),
        name=args.name,
        verbose=args.verbose,
    )

    best_model = project_path / args.name / "weights" / "best.pt"
    print(f"训练完成。最佳模型通常位于：{best_model}")
    print("将该文件复制到 models/mask_yolo.pt 后，本地图形化程序即可加载使用。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
