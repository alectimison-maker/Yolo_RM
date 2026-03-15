#!/usr/bin/env python3
"""
数据集整理脚本
─────────────────────────────────────────────────────────────────
功能:
  1. 只保留含有 13 列 pose 标签的图片（直接丢弃5列纯检测标注）
  2. 随机 8:2 分割 train/val 集
  3. 输出目录结构: {dst}/train/images、{dst}/train/labels 等

为什么只保留 13 列标签?
─────────────────────────────────────────────────────────────────
原始数据集中存在两种格式:

  【5 列格式 - 纯检测标注，丢弃】
    class  x_center  y_center  width  height
    原因: 关键点全部未知，若补零 (0,0) 会向模型传递错误的监督信号，
          导致同一类别出现"关键点在左上角"和"关键点在真实位置"的矛盾标注。

  【13 列格式 - 含关键点标注，保留】
    class  x_center  y_center  width  height  kp1x kp1y kp2x kp2y kp3x kp3y kp4x kp4y
    原因: 四角关键点均有真实坐标，可用于 pose 估计训练。

关键点含义 (4 个角点，顺时针顺序):
  kp1: 装甲板左上角
  kp2: 装甲板右上角
  kp3: 装甲板右下角
  kp4: 装甲板左下角

类别 ID 保持原始编号不变（0-13，共 14 类）。

用法:
  python prepare_dataset.py --src /path/to/raw_dataset --dst /path/to/output
"""
import random, shutil, logging, argparse
from pathlib import Path
from collections import Counter

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="原始数据集根目录，含 images/ 和 labels/ 子目录")
    p.add_argument("--dst", required=True, help="输出目录（自动创建）")
    p.add_argument("--val-ratio", type=float, default=0.2, help="验证集比例，默认 0.2 (8:2)")
    p.add_argument("--seed",      type=int,   default=42,  help="随机种子")
    return p.parse_args()

def filter_label(txt_path):
    """
    读取标签文件，只保留 13 列行，丢弃 5 列行。

    过滤规则:
      - 13 列 → 保留，类别 ID 不做任何修改
      - 5 列  → 丢弃（无关键点信息，避免补零引入矛盾监督信号）
      - 其他  → 丢弃（脏数据）

    Returns:
        list[str]: 过滤后的行列表，为空则该图片无有效标注
    """
    lines_out = []
    with open(txt_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 13:
                lines_out.append(line.strip())
            # 5列及其他列数直接跳过
    return lines_out

def main():
    args = parse_args()
    src = Path(args.src)
    dst = Path(args.dst)

    dst.mkdir(parents=True, exist_ok=True)
    log_file = dst / "prepare.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file)]
    )
    log = logging.getLogger()

    src_img = src / "images"
    src_lbl = src / "labels"
    assert src_img.exists(), f"找不到 images 目录: {src_img}"
    assert src_lbl.exists(), f"找不到 labels 目录: {src_lbl}"

    for split in ["train", "val"]:
        (dst / split / "images").mkdir(parents=True, exist_ok=True)
        (dst / split / "labels").mkdir(parents=True, exist_ok=True)

    # 筛选：只保留标签文件中至少有一行 13 列数据的图片
    img_files = sorted(src_img.glob("*.jpg"))
    log.info(f"原始图片总数: {len(img_files)}")

    valid, skipped = [], 0
    for img in img_files:
        lbl = src_lbl / (img.stem + ".txt")
        if not lbl.exists():
            skipped += 1
            continue
        if filter_label(lbl):   # 至少有一行 13 列标注
            valid.append(img)
        else:
            skipped += 1        # 纯 5 列或无标注，丢弃

    log.info(f"保留（含13列标注）: {len(valid)} 张")
    log.info(f"丢弃（纯5列/无标注）: {skipped} 张")

    # 随机 8:2 划分
    random.seed(args.seed)
    random.shuffle(valid)
    n_val   = max(1, int(len(valid) * args.val_ratio))
    n_train = len(valid) - n_val
    val_set, train_set = valid[:n_val], valid[n_val:]
    log.info(f"train: {n_train}, val: {n_val}")

    cls_counter = Counter()

    def copy_split(img_list, split_name):
        for i, img in enumerate(img_list):
            lbl = src_lbl / (img.stem + ".txt")
            lines = filter_label(lbl)
            shutil.copy2(img, dst / split_name / "images" / img.name)
            out_lbl = dst / split_name / "labels" / (img.stem + ".txt")
            out_lbl.write_text("\n".join(lines) + "\n")
            for line in lines:
                cls_counter[int(line.split()[0])] += 1
            if (i + 1) % 1000 == 0 or (i + 1) == len(img_list):
                log.info(f"  [{split_name}] {i+1}/{len(img_list)}")

    log.info("=== 处理 train ===")
    copy_split(train_set, "train")
    log.info("=== 处理 val ===")
    copy_split(val_set, "val")

    names = {0:'B1',1:'B2',2:'B3',3:'B4',4:'B5',5:'BO',6:'BS',
             7:'R1',8:'R2',9:'R3',10:'R4',11:'R5',12:'RO',13:'RS'}
    log.info("各类别实例数:")
    for c in sorted(cls_counter):
        log.info(f"  {c:2d} ({names.get(c,'?')}): {cls_counter[c]}")

    n_tr = len(list((dst / "train" / "images").glob("*.jpg")))
    n_va = len(list((dst / "val"   / "images").glob("*.jpg")))
    log.info(f"完成: train={n_tr}, val={n_va}")

if __name__ == "__main__":
    main()
