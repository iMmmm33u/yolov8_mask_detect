# Scripts

本目录存放项目辅助脚本。

## 检查 GPU 和依赖

```bash
python scripts/check_gpu.py
```

主要输出：

- `CUDA available: True`：当前环境可以使用 NVIDIA GPU 训练。
- `MPS available: True`：当前环境可以尝试 Apple Silicon MPS 加速。
- `Ultralytics version`：确认 YOLO 训练依赖可用。

## 训练模型

GPU 服务器推荐命令：

```bash
python scripts/train.py --device 0 --epochs 100 --imgsz 640 --batch 32 --workers 8
```

Apple Silicon 本地 MPS 命令：

```bash
python scripts/train.py --device mps --epochs 20 --imgsz 416 --batch 4 --workers 0
```

CPU 流程验证命令：

```bash
python scripts/train.py --device cpu --epochs 1 --imgsz 416 --batch 2 --workers 0
```

训练脚本默认读取：

```text
datasets/mask_dataset/mask.yaml
```

默认输出：

```text
runs/detect/mask_train/weights/best.pt
```

训练完成后，将最佳模型复制到：

```text
models/mask_yolo.pt
```

## 进度显示

`train.py` 已集成 `tqdm`：

- 训练前检查 train / val / test 图片和标注是否匹配。
- 训练时显示 epoch 总进度。

显示详细 Ultralytics 日志：

```bash
python scripts/train.py --device 0 --epochs 100 --imgsz 640 --batch 32 --workers 8 --verbose
```

关闭自定义进度条：

```bash
python scripts/train.py --device 0 --epochs 100 --imgsz 640 --batch 32 --workers 8 --no-progress
```

跳过数据检查：

```bash
python scripts/train.py --device 0 --epochs 100 --imgsz 640 --batch 32 --workers 8 --skip-data-check
```
