# RM YOLO11n-pose 装甲板关键点检测

基于 Ultralytics YOLO11n-pose，针对 RoboMaster 比赛装甲板检测与关键点定位任务的完整训练工程。

模型输出装甲板的检测框（14 类）和四角关键点坐标，关键点可直接用于 PnP 解算相机位姿。

---

## 项目结构

```
rm_yolo/
├── configs/
│   └── rm_dataset.yaml           # 数据集配置（类别定义、关键点形状、flip_idx）
├── data/                         # 数据目录
│   ├── dataset_raw/              # 原始 zip 解压后的数据（prepare_dataset.py 的输入）
│   │   └── XJTLU_2023_Detection_ALL/
│   │       ├── images/           # 所有原始图片（.jpg）
│   │       └── labels/           # 对应标签（5 列或 13 列混合格式）
│   ├── dataset/                  # prepare_dataset.py 输出（标签统一 + 8:2 划分）
│   │   ├── train/images/
│   │   ├── train/labels/
│   │   ├── val/images/
│   │   └── val/labels/
│   └── dataset_aug/              # augment_dataset.py 输出（HSV V 通道扩容后）
│       ├── train/images/         # ~50,724 张
│       ├── train/labels/
│       ├── val/images/           # 3,350 张（不扩容）
│       └── val/labels/
├── runs/                         # 训练输出（不上传 GitHub）
│   └── rm_pose_v1/
│       ├── weights/
│       │   ├── best.pt           # 验证集最优权重（val fitness 历史最高时保存）
│       │   ├── last.pt           # 最后一轮权重
│       │   └── epoch{N}.pt       # 每 25 轮检查点
│       ├── args.yaml             # 本次训练的完整参数记录
│       ├── results.csv           # 每轮详细指标（可用于绘图分析）
│       ├── results.png           # 训练曲线图（训练结束后生成）
│       ├── labels.jpg            # 数据集标签分布图
│       ├── train_batch*.jpg      # 训练 batch 可视化样本
│       ├── best.onnx             # ONNX 导出（训练结束后自动生成）
│       └── best_openvino_model/  # OpenVINO 导出（训练结束后自动生成）
├── prepare_dataset.py            # Step 1：标签格式统一 + 8:2 随机划分
├── augment_dataset.py            # Step 2：HSV V 通道离线扩容至约 10 万张
├── train.py                      # Step 3：训练 + 自动导出 ONNX/OpenVINO
├── run_pipeline.sh               # 一键运行完整流水线
├── yolo11n-pose.pt               # 预训练权重
└── README.md
```

> `data/`、`runs/`、`*.log`、`yolo11n-pose.pt` 均在 `.gitignore` 中，不会上传到仓库。

---

## 预训练权重说明（yolo11n-pose.pt）

`yolo11n-pose.pt` 是 Ultralytics 官方在 **COCO 数据集**上预训练的 YOLO11 nano pose 模型权重。它已经学习了大量通用视觉特征（边缘、纹理、形状等），以此为起点进行微调（迁移学习）可以大幅加速收敛，比从随机初始化训练节省数倍时间。

```bash
# 方法一：通过 ultralytics 自动下载（需联网）
python -c "from ultralytics import YOLO; YOLO('yolo11n-pose.pt')"
# 下载后复制到项目根目录即可

# 方法二：手动下载后放到项目根目录
# https://github.com/ultralytics/assets/releases 找 yolo11n-pose.pt
```

网络受限时（如实验室服务器），在本地下载后 scp 传到服务器：
```bash
scp yolo11n-pose.pt user@server:/path/to/rm_yolo/
```

---

## 快速开始

### 1. 环境准备

```bash
conda create -n rm_yolo python=3.10 -y
conda activate rm_yolo
pip install ultralytics opencv-python-headless
```

### 2. 准备原始数据集

原始数据集目录结构要求：

```
/your/raw_dataset/
├── images/
│   ├── 001.jpg
│   └── ...
└── labels/
    ├── 001.txt    # YOLO 格式，5 列（仅检测框）或 13 列（含关键点）均可
    └── ...
```

修改 `configs/rm_dataset.yaml` 中的 `path` 字段，指向扩容后数据集的根目录：

```yaml
path: /your/absolute/path/to/data/dataset_aug   # ← 改为实际路径
```

### 3. 一键运行

```bash
cd rm_yolo

# 单卡（GPU 0）
bash run_pipeline.sh --src-dataset /path/to/raw_dataset --device 0 --batch 64

# 多卡 DDP（GPU 0 + 1）
bash run_pipeline.sh --src-dataset /path/to/raw_dataset --device 0,1 --batch 128
```

流水线自动依次执行：数据整理 → 扩容 → 训练 → 导出 ONNX + OpenVINO。已完成的步骤会自动跳过。

---

## 分步运行

```bash
# Step 1：整理数据集（标签格式统一 + 8:2 划分）
python prepare_dataset.py \
    --src /path/to/raw_dataset \
    --dst ./data/dataset

# Step 2：HSV V 通道扩容（train 扩至约 10 万张，val 不变）
python augment_dataset.py \
    --src ./data/dataset \
    --dst ./data/dataset_aug \
    --target 100000

# Step 3：单卡训练
python train.py --device 0 --batch 64

# Step 3：多卡 DDP 训练
python train.py --device 0,1 --batch 128
```

---

## 数据集说明

### 原始数据规模

| 集合 | 图片数 |
|------|--------|
| 总计 | 16,753 |
| 训练集（80%） | 13,403 |
| 验证集（20%） | 3,350 |

划分方式：`random.seed(42)` 随机打乱后按 8:2 切分，结果可复现。

### 标签格式统一

原始数据存在两种格式混用，直接用于 YOLO Pose 训练会报错：

| 格式 | 列数 | 内容 |
|------|------|------|
| 纯检测标注 | 5 列 | `class x y w h` |
| Pose 标注 | 13 列 | `class x y w h kp1x kp1y kp2x kp2y kp3x kp3y kp4x kp4y` |

`prepare_dataset.py` 将 5 列格式补零关键点，统一为 13 列：
```
class x y w h  →  class x y w h 0 0 0 0 0 0 0 0
```

### 数据增强（两层叠加）

**第一层：离线扩容（augment_dataset.py，生成静态文件保存到磁盘）**

调整 HSV V（亮度）通道模拟不同曝光，每张原图生成 9 个版本：

```python
# V 系数均匀分布在 [0.4, 1.6]：从极暗到过曝
# v_scales 由 target/n 动态计算，target=50000, n=5637 → 9个版本
# v_scales = linspace(0.4, 1.6, 9)
# 转 HSV → 只缩放 V 通道 → 还原 BGR
hsv[:, :, 2] = np.clip(hsv[:, :, 2] * scale, 0, 255)
```

标签坐标不受亮度变化影响，直接复制。

**第二层：在线增强（训练时 YOLO 内置，每 batch 实时随机，不保存文件）**

| 增强项 | 值 | 说明 |
|--------|----|------|
| hsv_h/s/v | 0.015/0.7/0.4 | 色调/饱和度/亮度随机扰动 |
| degrees | 5.0° | 随机旋转 |
| translate | 10% | 随机平移 |
| scale | ±50% | 随机缩放（模拟不同距离） |
| fliplr | 50% | 水平翻转（flip_idx 保证关键点顺序正确） |
| mosaic | 50% | 4 张图拼接（增加小目标场景） |
| erasing | 40% | 随机擦除（模拟目标被遮挡） |

| 集合 | 数量 |
|------|------|
| 训练集（扩容后） | ~50,724 |
| 验证集 | 3,350（不扩容） |

### 标签格式（YOLO Pose，13 列）

```
class  x_center  y_center  width  height  kp1x  kp1y  kp2x  kp2y  kp3x  kp3y  kp4x  kp4y
```

4 个关键点为装甲板四角，顺序：**左上 → 右上 → 右下 → 左下**（顺时针）。

---

## 类别定义

| ID | 名称 | 说明 |
|----|------|------|
| 0  | B1 | 蓝方一号机器人 |
| 1  | B2 | 蓝方二号机器人 |
| 2  | B3 | 蓝方三号机器人 |
| 3  | B4 | 蓝方四号机器人 |
| 4  | B5 | 蓝方五号机器人 |
| 5  | BO | 蓝方前哨站 |
| 6  | BS | 蓝方哨兵（原始数据无关键点标注，关键点补零） |
| 7  | R1 | 红方一号机器人 |
| 8  | R2 | 红方二号机器人 |
| 9  | R3 | 红方三号机器人 |
| 10 | R4 | 红方四号机器人 |
| 11 | R5 | 红方五号机器人 |
| 12 | RO | 红方前哨站 |
| 13 | RS | 红方哨兵（原始数据无关键点标注，关键点补零） |

> **B3 样本极少**：labels.jpg 中 B3 数据量很少，这是原始数据集本身标注分布不均衡的问题，不影响其他类别的训练效果。

---

## 训练配置

### 模型

- 基础模型：YOLO11n-pose（nano 变体，最轻量）
- 参数量：2,664,594（~2.7M）
- 计算量：6.7 GFLOPs

### 关键点配置（configs/rm_dataset.yaml）

```yaml
kpt_shape: [4, 2]       # 4 个关键点，每个 (x, y)，无 visibility 维度
flip_idx: [1, 0, 3, 2]  # 水平翻转时关键点重映射：kp0↔kp1，kp2↔kp3
```

### 超参数（完整说明见 runs/rm_pose_v1/args.yaml）

| 参数 | 值 | 说明 |
|------|----|------|
| epochs | 300 | 最大训练轮数（早停可提前结束） |
| batch | 64/128 | 单卡 64，双卡 128（全局 batch size） |
| imgsz | 640 | 输入分辨率 |
| optimizer | AdamW | 自适应学习率优化器 |
| lr0 | 0.001 | 初始学习率 |
| lrf | 0.01 | 最终学习率系数（余弦退火到 lr0×lrf=0.00001） |
| weight_decay | 0.0005 | L2 正则化 |
| warmup_epochs | 3 | 学习率预热（避免初期梯度爆炸） |
| patience | 50 | 早停耐心值 |
| save_period | 25 | 每 25 轮保存检查点 |

---

## 早停机制

监控指标 fitness（Ultralytics 内置公式，不可配置）：

```
fitness = 0.1 × mAP50(B) + 0.9 × mAP50-95(B)
```

权重设计逻辑：mAP50-95 要求在 IoU 0.5~0.95 的多个严格阈值下都精准，比 mAP50 严格得多。用 0.9 的高权重驱动模型追求精确定位，而不仅仅是粗糙检测到目标。

每个 epoch 验证后：fitness 未创新高则计数器 +1；连续 50 轮无改善则停止。`best.pt` 始终保存 fitness 历史最优时刻的权重。

---

## 多卡训练原理（DDP）

```
batch=128 → GPU0 处理 64 张 + GPU1 处理 64 张（并行前向传播）
         → 各自反向传播 → All-Reduce 梯度平均 → 同步更新权重
```

数学上等价于单卡 batch=128，实际加速比约 **1.7×**。

---

## 训练产出文件格式说明

### best.pt（PyTorch 格式）

训练直接产出的权重文件，存储模型所有层的参数。

- 需要 Python + PyTorch + Ultralytics 环境
- 支持继续训练（resume）
- 不适合嵌入式部署

```python
from ultralytics import YOLO
model = YOLO("runs/rm_pose_v1/weights/best.pt")
results = model("test.jpg")
```

### best.onnx（ONNX 格式）

Open Neural Network Exchange，微软/Facebook 联合推出的跨框架标准格式。

- 跨平台（Windows/Linux/macOS）、跨语言（Python/C++/Java）
- 脱离 PyTorch 依赖，用 OpenCV DNN 或 ONNXRuntime 加载
- RM 视觉代码中最常用的部署格式

**适用场景：** 工控机、NUC、工业 PC 等通用 x86 平台

```cpp
// C++ ONNXRuntime 推理示例
Ort::Session session(env, "best.onnx", session_options);
```

### best_openvino_model/（OpenVINO 格式）

Intel 推出的推理优化框架，转换为 Intel 硬件专用中间表示（IR）。

输出两个文件：`best.xml`（网络结构）+ `best.bin`（权重数据）

- 在 Intel CPU 上推理速度比 PyTorch 快 2-5×
- 支持 Intel 神经计算棒（NCS2 / Myriad X VPU）
- 不依赖 GPU，适合无独立显卡的上位机

**适用场景：** RM 上位机（i7/i9 CPU），或 Intel NCS2

### 格式对比

| 格式 | 运行环境 | 优势 | 适用场景 |
|------|---------|------|---------|
| .pt | Python + CUDA | 灵活，支持继续训练 | 开发调试 |
| .onnx | 通用（CPU/GPU） | 跨平台跨语言 | C++ 部署、跨平台 |
| OpenVINO | Intel 硬件 | Intel CPU 最快 | RM 上位机无显卡 |

---

## 指标说明

| 指标 | 含义 |
|------|------|
| box_loss | 检测框位置误差（CIoU） |
| pose_loss | 关键点坐标误差（OKS） |
| cls_loss | 分类误差（交叉熵） |
| mAP@0.5 | IoU>0.5 时的平均精度（主要评估指标） |
| mAP@0.5:0.95 | IoU 0.5→0.95 均值（更严格） |
| (B) 后缀 | 目标检测指标 |
| (P) 后缀 | 关键点估计指标 |

---



## 数据集下载

完整数据集通过 GitHub Release 提供：

| 文件 | 说明 | 大小 |
|------|------|------|
| `dataset_raw.zip` | 原始标注数据（16,753 张图片） | ~1.2G |
| `dataset.zip` | 整理后 8:2 划分数据集 | ~683M |

下载后解压到 `data/` 目录：
```bash
unzip dataset_raw.zip -d data/dataset_raw
unzip dataset.zip -d data/dataset
```

> `dataset_aug/`（107k 张扩容数据集）可通过 `augment_dataset.py` 从 `dataset/` 生成，不单独分发。

---

## 预训练模型权重

| 文件 | 格式 | 说明 |
|------|------|------|
| `runs/rm_pose_v1/weights/best.pt` | PyTorch | 验证集最优权重 |
| `runs/rm_pose_v1/weights/best.onnx` | ONNX | 跨平台部署格式 |
| `runs/rm_pose_v1/weights/best_openvino_model/` | OpenVINO | Intel 硬件优化 |

---

## 训练结果参考

> 实际训练：epoch 158（设定上限 300，patience=50 提前停止），耗时约 16.8 小时（Tesla V100-PCIE-16GB）

| 指标 | 值 |
|------|-----|
| Box mAP@0.5 | 0.965 |
| Box mAP@0.5:0.95 | 0.713 |
| Pose mAP@0.5 | 0.917 |
| Pose mAP@0.5:0.95 | 0.888 |

各类别详细指标：

| 类别 | Box P | Box R | Box mAP50 | Pose P | Pose R | Pose mAP50 |
|------|-------|-------|-----------|--------|--------|------------|
| B1   | 0.940 | 0.930 | 0.961 | 0.898 | 0.892 | 0.903 |
| B2   | 0.959 | 0.955 | 0.973 | 0.932 | 0.936 | 0.944 |
| B3   | 0.946 | 0.878 | 0.929 | 0.943 | 0.878 | 0.931 |
| B4   | 0.951 | 0.954 | 0.978 | 0.923 | 0.930 | 0.936 |
| B5   | 0.971 | 0.876 | 0.942 | 0.924 | 0.848 | 0.886 |
| BO   | 1.000 | 0.974 | 0.995 | 1.000 | 0.991 | 0.995 |
| R1   | 0.986 | 0.969 | 0.986 | 0.879 | 0.865 | 0.826 |
| R2   | 0.955 | 0.884 | 0.948 | 0.917 | 0.853 | 0.893 |
| R3   | 0.923 | 0.958 | 0.953 | 0.884 | 0.926 | 0.916 |
| R4   | 0.956 | 0.924 | 0.979 | 0.930 | 0.902 | 0.935 |
| R5   | 0.946 | 0.922 | 0.971 | 0.932 | 0.918 | 0.958 |
| RO   | 0.953 | 0.953 | 0.971 | 0.888 | 0.894 | 0.875 |
