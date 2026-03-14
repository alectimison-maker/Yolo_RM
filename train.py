#!/usr/bin/env python3
"""
RM YOLO11n-pose 训练脚本
─────────────────────────────────────────────────────────────────
训练逻辑概述:
  1. 加载 yolo11n-pose.pt 预训练权重（迁移学习起点）
  2. 在 RM 装甲板数据集上微调，输出检测框 + 4 个角点关键点
  3. 训练结束后自动导出 ONNX 和 OpenVINO 格式用于部署

用法:
  python train.py --device 0          # 单卡 GPU 0
  python train.py --device 0,1        # 双卡 DDP 训练
  python train.py --device 0 --batch 64
"""
import logging, sys, argparse
from pathlib import Path
from ultralytics import YOLO

# 项目根目录：以本脚本所在目录为准，无需硬编码路径
PROJECT = Path(__file__).parent

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="0",    help="GPU 编号，多卡用逗号分隔如 '0,1'")
    p.add_argument("--batch",  type=int, default=64, help="全局 batch size，单卡建议 64，双卡建议 128")
    p.add_argument("--epochs", type=int, default=300, help="最大训练轮数（早停可提前结束）")
    p.add_argument("--data",   default=str(PROJECT / "configs" / "rm_dataset.yaml"),
                   help="数据集配置文件路径")
    p.add_argument("--name",   default="rm_pose_v1", help="训练输出目录名（runs/{name}/）")
    return p.parse_args()

def main():
    args = parse_args()

    # 日志同时写到终端和 run.log 文件
    log_file = PROJECT / "run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file)]
    )
    log = logging.getLogger("train")

    log.info("=" * 60)
    log.info("RM YOLO11n-pose 训练启动")
    log.info(f"GPU: {args.device} | batch={args.batch} | epochs={args.epochs}")
    log.info("=" * 60)

    # ══════════════════════════════════════════════════════════
    # 训练超参数配置
    # ══════════════════════════════════════════════════════════
    TRAIN_CFG = dict(
        data   = args.data,       # 数据集 yaml 路径
        epochs = args.epochs,     # 最大 epoch 数
        imgsz  = 640,             # 输入图像分辨率（正方形裁剪到 640×640）
        batch  = args.batch,      # 全局 batch size（DDP 时每卡 = batch/GPU数）
        device = args.device,     # 单卡: "0"，双卡 DDP: "0,1"
        workers = 8,              # DataLoader 子进程数，根据 CPU 核心数调整

        # ── 优化器 ────────────────────────────────────────────
        # AdamW = Adam + 解耦权重衰减，相比 SGD 在小数据集上收敛更快
        optimizer    = "AdamW",
        lr0          = 0.001,     # 初始学习率
        lrf          = 0.01,      # 最终学习率 = lr0 × lrf，余弦退火到此值
        momentum     = 0.937,     # AdamW 的 β1 参数（梯度一阶矩衰减系数）
        weight_decay = 0.0005,    # L2 正则化系数，防止权重过大导致过拟合
        warmup_epochs = 3,        # 前 3 个 epoch 学习率从 0 线性升到 lr0（避免初期梯度爆炸）

        # ── 早停 ──────────────────────────────────────────────
        # 监控指标: fitness = 0.1×mAP50(B) + 0.9×mAP50-95(B)
        # 连续 patience 个 epoch 无改善则停止，best.pt 保存最优时刻的权重
        patience = 50,

        # ── 在线数据增强（每个 batch 实时随机生成，不保存文件）──
        # 与扩容阶段的 V 通道离线增强叠加，模型实际看到双重增强效果
        hsv_h     = 0.015,  # 色调随机偏移 ±1.5%（H 通道，轻微变色）
        hsv_s     = 0.7,    # 饱和度随机缩放 [0.3, 1.7]（模拟灯光变化）
        hsv_v     = 0.4,    # 亮度随机缩放 [0.6, 1.4]（与离线扩容互补）
        degrees   = 5.0,    # 随机旋转 ±5°（装甲板不会大角度倾斜）
        translate = 0.1,    # 随机平移 ±10% 图像宽高
        scale     = 0.5,    # 随机缩放 [0.5, 1.5]（模拟不同距离）
        fliplr    = 0.5,    # 水平翻转概率 50%（flip_idx 保证关键点顺序正确）
        mosaic    = 0.5,    # Mosaic 拼接概率 50%（4 张图拼一张，增加小目标）

        # ── 输出控制 ──────────────────────────────────────────
        save_period = 25,   # 每 25 epoch 保存一次检查点（weights/epoch{N}.pt）
        project  = str(PROJECT / "runs"),  # 输出根目录
        name     = args.name,             # 本次训练子目录名
        exist_ok = True,    # 目录已存在时继续写入（不报错）
        plots    = True,    # 训练结束后生成 results.png 等可视化图表
        val      = True,    # 每个 epoch 结束后在 val 集上评估
        cache    = False,   # 不缓存图片到内存（数据量大时节省内存）
    )

    # ══════════════════════════════════════════════════════════
    # 加载预训练权重（迁移学习）
    # ══════════════════════════════════════════════════════════
    # yolo11n-pose.pt 是 Ultralytics 在 COCO 数据集上预训练的权重
    # 已学习到边缘、纹理、形状等通用特征，以此为起点大幅加速收敛
    # 优先使用项目目录下的本地文件（网络受限环境），否则触发自动下载
    pretrain = PROJECT / "yolo11n-pose.pt"
    model_src = str(pretrain) if pretrain.exists() else "yolo11n-pose.pt"
    log.info(f"加载预训练权重: {model_src}")
    model = YOLO(model_src)

    # ══════════════════════════════════════════════════════════
    # 启动训练
    # 多卡时 Ultralytics 自动启用 DDP，fork 多个进程分别绑定各 GPU：
    #   - 每个 batch 按 GPU 数平均切分数据
    #   - 各 GPU 独立前向传播 + 反向传播
    #   - All-Reduce 平均梯度后同步更新权重
    # ══════════════════════════════════════════════════════════
    model.train(**TRAIN_CFG)
    log.info("训练完成")

    # ══════════════════════════════════════════════════════════
    # 导出部署格式
    # 训练产出的 best.pt 是 PyTorch 格式，需要完整的 Python 环境才能运行。
    # 导出为通用格式以便在嵌入式/跨平台环境部署（详见 README）。
    # ══════════════════════════════════════════════════════════
    best_pt = PROJECT / "runs" / args.name / "weights" / "best.pt"
    if not best_pt.exists():
        log.error(f"未找到最佳权重: {best_pt}")
        return

    log.info(f"最佳权重: {best_pt}")
    best_model = YOLO(str(best_pt))

    log.info("导出 ONNX（跨平台推理格式）...")
    best_model.export(format="onnx", imgsz=640, opset=12, simplify=True)

    log.info("导出 OpenVINO（Intel 硬件优化格式）...")
    best_model.export(format="openvino", imgsz=640)

    log.info("=== 输出文件 ===")
    for f in sorted((PROJECT / "runs" / args.name).rglob("*")):
        if f.suffix in {".pt", ".onnx", ".xml", ".bin"}:
            log.info(f"  {f.relative_to(PROJECT)}")
    log.info("全部完成！")

if __name__ == "__main__":
    main()
