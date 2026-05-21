import os
import platform


if platform.system() == "Darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def main() -> int:
    try:
        import torch
    except ImportError:
        print("PyTorch 未安装。请先安装项目依赖。")
        return 1

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for index in range(torch.cuda.device_count()):
            print(f"GPU {index}: {torch.cuda.get_device_name(index)}")
    else:
        print("当前环境未检测到可用 GPU。可以用于本地开发或 CPU 流程验证，不建议正式训练。")

    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    print(f"MPS available: {mps_available}")
    if mps_available:
        print("当前环境检测到 Apple Silicon MPS，可尝试使用 --device mps 进行本地加速。")

    try:
        import ultralytics

        print(f"Ultralytics version: {ultralytics.__version__}")
    except ImportError:
        print("Ultralytics 未安装。请先安装项目依赖。")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
