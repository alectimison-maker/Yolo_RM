#!/usr/bin/env python3
"""
数据集扩容脚本 —— HSV V 通道曝光模拟
─────────────────────────────────────────────────────────────────
扩容策略:
  针对海康威视相机在 RM 比赛中曝光值变化大的特点（曝光值约 5000），
  通过调整图像 HSV 颜色空间的 V（亮度）通道来模拟不同曝光效果，
  使模型对亮/暗环境均具备鲁棒性。

  每张原始图生成 8 个曝光版本，V 缩放系数均匀分布在 [0.4, 1.6]：
    0.4  → 极暗（曝光严重不足）
    0.57 → 较暗
    0.74 → 偏暗
    0.91 → 接近正常
    1.09 → 接近正常（略亮）
    1.26 → 偏亮
    1.43 → 较亮
    1.6  → 极亮（过曝）

扩容范围:
  - train 集: 原始图 × 8，约 13,403 × 8 ≈ 107,000 张
  - val   集: 原样复制，不做扩容（保持验证集的真实分布）

注意: 扩容只改变图像的亮度，标签文件（关键点坐标）完全不变，
      直接复制到对应目录即可。

用法:
  python augment_dataset.py --src ./data/dataset --dst ./data/dataset_aug --target 100000
"""
import cv2, shutil, logging, math, argparse
import numpy as np
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src",    required=True, help="输入数据集目录（含 train/val 子目录）")
    p.add_argument("--dst",    required=True, help="扩容后输出目录（自动创建）")
    p.add_argument("--target", type=int, default=100000, help="目标 train 图片数，默认 100000")
    return p.parse_args()

# ══════════════════════════════════════════════════════════════
# 核心扩容函数：HSV V 通道亮度缩放
# ══════════════════════════════════════════════════════════════
def adjust_v(img_bgr, scale):
    """
    调整图像 V（亮度）通道，模拟不同曝光效果。

    原理:
      OpenCV 默认使用 BGR 色彩空间，直接修改亮度会同时改变色相和饱和度。
      转换到 HSV 后，H（色相）、S（饱和度）、V（亮度）相互独立，
      只缩放 V 通道不会影响颜色，仅改变整体明暗。

    操作步骤:
      BGR → HSV（float32）→ V 通道 × scale → clip 到 [0, 255] → uint8 → BGR

    Args:
        img_bgr: OpenCV 读取的 BGR 图像 (H, W, 3)
        scale:   亮度缩放系数，<1 变暗，>1 变亮

    Returns:
        调整后的 BGR 图像
    """
    # 转 HSV 并升为 float32，避免 uint8 乘法溢出
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    # 只修改 V 通道（索引 2），H 和 S 保持不变
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * scale, 0, 255)
    # 转回 uint8 并还原为 BGR
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
# ══════════════════════════════════════════════════════════════

def augment(src_dir, dst_dir, target=100000):
    src, dst = Path(src_dir), Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)

    log_file = dst / "augment.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file)]
    )
    log = logging.getLogger()

    train_imgs = sorted((src / "train" / "images").glob("*.jpg"))
    val_imgs   = sorted((src / "val"   / "images").glob("*.jpg"))
    n = len(train_imgs)
    log.info(f"原始 train={n}, val={len(val_imgs)}")

    # ── 计算每张原图需要生成几个版本 ──────────────────────────
    # repeats = ceil(target / n)，使总数≥target
    repeats  = max(1, math.ceil(target / max(n, 1)))
    # V 缩放系数均匀分布在 [0.4, 1.6] 区间
    v_scales = [round(0.4 + i * 1.2 / max(repeats - 1, 1), 3) for i in range(repeats)]
    log.info(f"每张生成 {repeats} 个版本, 实际 train≈{n * repeats}, V 范围 [{v_scales[0]}, {v_scales[-1]}]")

    for split in ["train", "val"]:
        (dst / split / "images").mkdir(parents=True, exist_ok=True)
        (dst / split / "labels").mkdir(parents=True, exist_ok=True)

    def process(img_list, split, do_aug):
        """
        处理一个数据集分组。

        do_aug=True  (train): 对每张图生成多个不同亮度版本
        do_aug=False (val):   原图原样复制，保持验证集真实分布
        """
        scales = v_scales if do_aug else [1.0]  # val 集 scale=1.0 即不改变
        for i, img_p in enumerate(img_list):
            img = cv2.imread(str(img_p))
            if img is None:
                log.warning(f"跳过损坏图片: {img_p}")
                continue
            lbl_p = src / split / "labels" / (img_p.stem + ".txt")
            for j, sc in enumerate(scales):
                # 命名规则: 原始图保持原名，增强版追加 _v{scale}
                stem = img_p.stem if (sc == 1.0 and j == 0) \
                       else f"{img_p.stem}_v{str(sc).replace('.', 'p')}"
                out_img = dst / split / "images" / (stem + ".jpg")
                out_lbl = dst / split / "labels" / (stem + ".txt")
                # 写出增强后的图片（JPEG 质量 95 保留足够细节）
                cv2.imwrite(str(out_img), adjust_v(img, sc), [cv2.IMWRITE_JPEG_QUALITY, 95])
                # 标签坐标不受亮度变化影响，直接复制
                if lbl_p.exists():
                    shutil.copy2(str(lbl_p), str(out_lbl))
                else:
                    out_lbl.touch()  # 无标签则创建空文件（背景图）
            if (i + 1) % 1000 == 0 or (i + 1) == len(img_list):
                log.info(f"  [{split}] {i+1}/{len(img_list)}")

    log.info("=== 扩容 train ===")
    process(train_imgs, "train", True)
    log.info("=== 复制 val（不增强）===")
    process(val_imgs, "val", False)

    n_tr = len(list((dst / "train" / "images").glob("*.jpg")))
    n_va = len(list((dst / "val"   / "images").glob("*.jpg")))
    log.info(f"完成: train={n_tr}, val={n_va}")

if __name__ == "__main__":
    args = parse_args()
    augment(args.src, args.dst, args.target)
