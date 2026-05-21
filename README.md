# 基于 YOLO 的口罩检测系统

这是一个基于 Python、Ultralytics YOLO 和 PyQt5 的口罩检测系统。项目支持训练自定义口罩检测模型，并通过图形化界面选择本地图片，识别 `mask` 和 `no_mask` 两类目标，最后根据检测结果判断口罩佩戴情况。

当前已实现功能：

- 图片口罩检测
- `mask` / `no_mask` 两类目标识别
- 检测框与置信度绘制
- 检测结果明细展示
- 口罩佩戴情况汇总判断
- MPS / CUDA / CPU 训练入口
- 训练过程 `tqdm` 进度显示

## 项目结构

```text
mask_detect/
├── data/
│   └── mask-dataset/              # 原始下载数据，可保留
├── datasets/
│   └── mask_dataset/              # YOLO 训练数据
│       ├── images/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       ├── labels/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       └── mask.yaml
├── models/
│   └── mask_yolo.pt               # 训练完成后放置的模型文件
├── scripts/
│   ├── README.md
│   ├── check_gpu.py
│   └── train.py
├── src/
│   ├── config.py
│   ├── detector.py
│   ├── main.py
│   ├── ui.py
│   └── utils.py
├── environment.yml
├── requirements.txt
└── README.md
```

## 环境安装

建议使用 Python 3.10。

本地开发环境：

```bash
conda env create -f environment.yml
conda activate mask-detect
```

如果需要手动创建环境：

```bash
conda create -n mask-detect python=3.10 -y
conda activate mask-detect
pip install -r requirements.txt
```

服务器训练环境可以更轻量，不需要安装 GUI 相关依赖：

```bash
conda create -n mask-train python=3.10 -y
conda activate mask-train
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install ultralytics opencv-python Pillow numpy tqdm
```

国内服务器如果 PyPI 较慢，可以给第二条 `pip` 命令加镜像源：

```bash
pip install ultralytics opencv-python Pillow numpy tqdm -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 数据集下载与准备

本项目使用的数据集为 Kaggle 上的 Face Mask Detection YOLO/Darknet 格式数据集：

```text
https://www.kaggle.com/datasets/parot99/face-mask-detection-yolo-darknet-format?resource=download
```

下载后可将压缩包解压到项目的 `data/` 目录，建议保留为：

```text
data/mask-dataset/
├── images/
│   ├── train/
│   ├── validate/
│   └── test/
├── train.txt
├── validate.txt
└── test.txt
```

该数据集原始目录中，图片和同名 `.txt` 标注文件位于同一个 split 目录下。项目训练脚本使用标准 YOLO 目录，因此需要整理为：

```text
datasets/mask_dataset/
├── images/train/
├── images/val/
├── images/test/
├── labels/train/
├── labels/val/
└── labels/test/
```

可以在项目根目录执行以下命令整理数据：

```bash
mkdir -p datasets/mask_dataset/images/train datasets/mask_dataset/images/val datasets/mask_dataset/images/test
mkdir -p datasets/mask_dataset/labels/train datasets/mask_dataset/labels/val datasets/mask_dataset/labels/test

find data/mask-dataset/images/train -maxdepth 1 -type f \( -name '*.jpg' -o -name '*.jpeg' -o -name '*.png' -o -name '*.bmp' -o -name '*.webp' \) -exec cp {} datasets/mask_dataset/images/train/ \;
find data/mask-dataset/images/validate -maxdepth 1 -type f \( -name '*.jpg' -o -name '*.jpeg' -o -name '*.png' -o -name '*.bmp' -o -name '*.webp' \) -exec cp {} datasets/mask_dataset/images/val/ \;
find data/mask-dataset/images/test -maxdepth 1 -type f \( -name '*.jpg' -o -name '*.jpeg' -o -name '*.png' -o -name '*.bmp' -o -name '*.webp' \) -exec cp {} datasets/mask_dataset/images/test/ \;

find data/mask-dataset/images/train -maxdepth 1 -type f -name '*.txt' -exec cp {} datasets/mask_dataset/labels/train/ \;
find data/mask-dataset/images/validate -maxdepth 1 -type f -name '*.txt' -exec cp {} datasets/mask_dataset/labels/val/ \;
find data/mask-dataset/images/test -maxdepth 1 -type f -name '*.txt' -exec cp {} datasets/mask_dataset/labels/test/ \;
```

整理完成后，训练集、验证集和测试集的图片数量应分别与标注数量一致。训练脚本会在启动时自动检查这一点。

类别定义以 [datasets/mask_dataset/mask.yaml](datasets/mask_dataset/mask.yaml) 为准：

```text
0 no_mask
1 mask
```

注意：该 Kaggle 数据集的 `classes.txt` 顺序是 `no-mask`、`mask`，因此本项目的 `mask.yaml` 使用 `0 no_mask`、`1 mask`。如果类别顺序写反，模型推理结果会把戴口罩和未戴口罩识别反。

每张图片对应一个同名 `.txt` 标注文件，格式为：

```text
class_id center_x center_y width height
```

坐标均为 YOLO 标准归一化坐标。

## 检查训练设备

```bash
python scripts/check_gpu.py
```

输出含义：

- `CUDA available: True`：可使用 NVIDIA GPU 训练。
- `MPS available: True`：可在 Apple Silicon Mac 上尝试 MPS 加速。
- 二者都为 `False`：只能 CPU 训练，适合流程验证，不适合完整训练。

## 本地训练

Apple Silicon Mac 可使用 MPS：

```bash
python scripts/train.py --device mps --epochs 20 --imgsz 416 --batch 4 --workers 0
```

如果 MPS 算子兼容性报错，可以尝试：

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/train.py --device mps --epochs 20 --imgsz 416 --batch 4 --workers 0
```

CPU 只建议用于流程验证：

```bash
python scripts/train.py --device cpu --epochs 1 --imgsz 416 --batch 2 --workers 0
```

## GPU 服务器训练

将项目上传到服务器后，进入项目目录：

```bash
cd ~/mask_detect
```

检查 GPU：

```bash
nvidia-smi
python scripts/check_gpu.py
```

RTX 4090 等大显存 GPU 推荐：

```bash
python scripts/train.py --device 0 --epochs 100 --imgsz 640 --batch 32 --workers 8
```

如果显存不足，将 `batch` 降低到 16 或 8：

```bash
python scripts/train.py --device 0 --epochs 100 --imgsz 640 --batch 16 --workers 8
```

训练脚本默认输出：

```text
runs/detect/mask_train/weights/best.pt
```

训练脚本会使用项目根目录解析 `--data` 和 `--project`，避免出现 `runs/detect/runs/detect` 这类嵌套输出。

## 训练进度参数

`scripts/train.py` 已集成 `tqdm`：

- 训练前会检查 train / val / test 图片和标注是否匹配。
- 训练时会显示 epoch 总进度。

常用参数：

```bash
python scripts/train.py --device 0 --epochs 100 --imgsz 640 --batch 32 --workers 8
```

显示 Ultralytics 详细日志：

```bash
python scripts/train.py --device 0 --epochs 100 --imgsz 640 --batch 32 --workers 8 --verbose
```

关闭自定义进度条：

```bash
python scripts/train.py --device 0 --epochs 100 --imgsz 640 --batch 32 --workers 8 --no-progress
```

跳过训练前数据检查：

```bash
python scripts/train.py --device 0 --epochs 100 --imgsz 640 --batch 32 --workers 8 --skip-data-check
```

## 预训练权重

如果不想自行训练，可以直接使用已经训练好的权重文件。推荐将权重文件作为 GitHub Release 附件提供，不直接提交到 Git 仓库。

下载权重后，将文件放到项目默认模型路径：

```text
models/mask_yolo.pt
```

本地可用以下命令检查文件是否存在：

```bash
ls -lh models/mask_yolo.pt
```

如果该文件存在，启动 GUI 后点击“开始检测”时会自动加载该模型。

## 下载服务器模型到本地

训练完成后，将服务器上的最佳权重下载到本地默认模型路径。

普通 SSH 端口：

```bash
scp root@服务器IP:~/mask_detect/runs/detect/mask_train/weights/best.pt /Users/wcx/mask_detect/models/mask_yolo.pt
```

自定义 SSH 端口，例如 AutoDL/SeetaCloud 给出的端口为 `36800`：

```bash
scp -P 36800 root@connect.westd.seetacloud.com:~/mask_detect/runs/detect/mask_train/weights/best.pt /Users/wcx/mask_detect/models/mask_yolo.pt
```

注意：`ssh` 使用小写 `-p`，`scp` 指定端口使用大写 `-P`。

下载后检查：

```bash
ls -lh /Users/wcx/mask_detect/models/mask_yolo.pt
```

## 启动程序

本地启动 GUI：

```bash
cd /Users/wcx/mask_detect
conda activate mask-detect
python src/main.py
```

如果 `models/mask_yolo.pt` 不存在，程序仍可启动，但点击检测时会提示模型文件缺失。

macOS 上可能出现类似日志：

```text
error messaging the mach port for IMKCFRunLoopWakeUpReliable
```

这是输入法框架日志，不影响程序运行。

## 口罩佩戴情况判断

程序会根据检测结果在结果区域展示汇总结论：

```text
佩戴情况：全部已佩戴口罩（已佩戴 2 人，未佩戴 0 人）。
```

```text
佩戴情况：存在未佩戴口罩人员（已佩戴 0 人，未佩戴 2 人）。
```

```text
佩戴情况：部分人员未佩戴口罩（已佩戴 1 人，未佩戴 1 人）。
```

判断逻辑位于 [src/utils.py](src/utils.py)。

## 调整检测阈值

默认置信度阈值位于 [src/config.py](src/config.py)：

```python
DEFAULT_CONFIDENCE = 0.25
```

如果误检较多，可以调高到 `0.4`；如果漏检较多，可以保持 `0.25` 或降低到 `0.2`。

## 后续优化方向

1. 增加检测结果图片保存功能。
2. 增加摄像头实时检测功能。
3. 增加 GUI 置信度滑块。
4. 增加检测结果导出功能。
5. 训练更高精度模型，例如 `yolov8s.pt`。
