#!/bin/bash
# RM YOLO11n-pose 完整训练流水线
# 用法: bash run_pipeline.sh --src-dataset /path/to/raw_dataset [--device 0] [--batch 64]
#
# 路径说明:
#   脚本默认以自身所在目录为项目根目录。
#   --src-dataset  原始数据集根目录（含 images/ 和 labels/ 子目录，首次运行必填）
#   --device       训练 GPU 编号，多卡用逗号分隔，如 '0,1'（默认 0）
#   --batch        全局 batch size（默认 64，多卡建议 128）
set -e

BASE="$(cd "$(dirname "$0")" && pwd)"

SRC_DATASET=""
DEVICE="0"
BATCH=64

while [[ $# -gt 0 ]]; do
    case "$1" in
        --src-dataset) SRC_DATASET="$2"; shift 2 ;;
        --device)      DEVICE="$2";      shift 2 ;;
        --batch)       BATCH="$2";       shift 2 ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

# 自动查找 Python
if command -v python3 &>/dev/null; then
    PYTHON=python3
elif command -v python &>/dev/null; then
    PYTHON=python
else
    echo "错误: 找不到 python，请激活对应 conda 环境后重试"
    exit 1
fi

LOG=$BASE/run.log
ts() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

ts "==== RM YOLO11n-pose 训练流水线启动 ====" | tee -a "$LOG"
ts "GPU: $DEVICE | batch: $BATCH" | tee -a "$LOG"

# Step 1: 整理数据集
ts "Step 1: 整理数据集..." | tee -a "$LOG"
if [ ! -d "$BASE/data/dataset/train" ]; then
    if [ -z "$SRC_DATASET" ]; then
        echo "错误: 首次运行需指定 --src-dataset /path/to/raw_dataset"
        echo "  raw_dataset 目录需包含 images/ 和 labels/ 子目录"
        exit 1
    fi
    $PYTHON "$BASE/prepare_dataset.py" \
        --src "$SRC_DATASET" \
        --dst "$BASE/data/dataset" 2>&1 | tee -a "$LOG"
else
    ts "数据集已存在，跳过 Step 1" | tee -a "$LOG"
fi

# Step 2: HSV-V 扩容
ts "Step 2: 数据扩容（target=50000，约 5 万张）..." | tee -a "$LOG"
if [ ! -d "$BASE/data/dataset_aug/train" ]; then
    $PYTHON "$BASE/augment_dataset.py" \
        --src "$BASE/data/dataset" \
        --dst "$BASE/data/dataset_aug" \
        --target 50000 2>&1 | tee -a "$LOG"
else
    ts "扩容数据集已存在，跳过 Step 2" | tee -a "$LOG"
fi

# Step 3: 训练
ts "Step 3: 开始训练 YOLO11n-pose..." | tee -a "$LOG"
$PYTHON "$BASE/train.py" --device "$DEVICE" --batch "$BATCH" 2>&1 | tee -a "$LOG"

ts "==== 流水线完成 ====" | tee -a "$LOG"
