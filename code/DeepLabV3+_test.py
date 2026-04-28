#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from scipy.ndimage import binary_closing, binary_opening
from skimage.morphology import remove_small_objects

import segmentation_models_pytorch as smp


@dataclass
class EvalConfig:
    # DeepLabV3+ 训练输出 best.pth
    model_path: str = r"D:\segformer-1.8\logs-deeplabv3plus\exp2\best.pth"

    # 数据集根目录（与你训练一致）
    test_dataset_path: str = r"D:\segformer-1.8\segformer-1.8\数据集\全分割16类V4.7\划分后数据集-全分割16类V4.7"
    img_dir: str = "JPEGImages"
    mask_dir: str = "SegmentationClass"
    split_subdir: str = r"ImageSets\Segmentation"
    test_list_name: str = "test.txt"

    input_size: int = 1024
    num_classes: int = 17
    ignore_index: int = 255
    use_amp: bool = True

    # 输出目录（会在 output_root 下生成一个带时间戳的子文件夹）
    output_root: str = "test_evaluation_results"

    # 后处理（和你 U-Net 测试脚本一致）
    enable_postprocessing: bool = True
    postprocess_config: dict = None

    # 类别名（长度需= num_classes）
    class_names: List[str] = None

    # DeepLabV3+ 结构参数（默认从 train_config.json 读取；读不到则用默认）
    encoder_name: str = "resnet50"
    encoder_weights: str = "imagenet"  # 这里仅用于记录，不影响评估加载（评估加载的是 best.pth）
    decoder_channels: int = 256


def ensure_class_names(cfg: EvalConfig):
    if not cfg.class_names or len(cfg.class_names) != cfg.num_classes:
        cfg.class_names = [f"class_{i}" for i in range(cfg.num_classes)]


def read_test_ids(test_txt_path: str) -> List[str]:
    with open(test_txt_path, "r", encoding="utf-8-sig") as f:
        return [line.strip() for line in f if line.strip()]


def load_rgb_image(path: str, size: int) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    img = img.resize((size, size), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr.transpose(2, 0, 1)  # [3,H,W]


def load_mask(path: str, size: int, num_classes: int) -> np.ndarray:
    m = Image.open(path)
    if m.mode == "P":
        arr = np.array(m, dtype=np.uint8)
    else:
        arr = np.array(m.convert("L"), dtype=np.uint8)
    arr = cv2.resize(arr, (size, size), interpolation=cv2.INTER_NEAREST)

    # 防御：如果 mask 里有异常 id（>=num_classes），统统压到 0（背景）
    arr = np.where(arr >= num_classes, 0, arr).astype(np.uint8)
    return arr


def conf_to_metrics(cm: np.ndarray):
    eps = 1e-6
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0).astype(np.float64) - tp
    fn = cm.sum(axis=1).astype(np.float64) - tp
    iou = tp / (tp + fp + fn + eps)
    dice = (2 * tp) / (2 * tp + fp + fn + eps)
    return iou, dice


def postprocess_prediction(pred: np.ndarray, post_cfg: dict) -> np.ndarray:
    out = pred.copy()
    for _, cfg in (post_cfg or {}).items():
        class_id = int(cfg.get("class_id", -1))
        if class_id < 0:
            continue
        class_mask = (pred == class_id).astype(bool)
        if not class_mask.any():
            continue
        close_it = int(cfg.get("closing_iterations", 0))
        open_it = int(cfg.get("opening_iterations", 0))
        min_area = int(cfg.get("min_area", 0))
        if close_it > 0:
            class_mask = binary_closing(class_mask, iterations=close_it)
        if open_it > 0:
            class_mask = binary_opening(class_mask, iterations=open_it)
        if min_area > 0:
            class_mask = remove_small_objects(class_mask.astype(bool), min_size=min_area)
        out[pred == class_id] = 0
        out[class_mask] = class_id
    return out


def try_load_train_config(cfg: EvalConfig):
    """
    从 best.pth 同目录的 train_config.json 读取 encoder_name / encoder_weights / decoder_channels（如果存在）
    """
    train_cfg_path = Path(cfg.model_path).resolve().parent / "train_config.json"
    if train_cfg_path.exists():
        try:
            with open(train_cfg_path, "r", encoding="utf-8") as f:
                tj = json.load(f)
            cfg.encoder_name = tj.get("encoder_name", cfg.encoder_name)
            cfg.encoder_weights = tj.get("encoder_weights", cfg.encoder_weights)
            cfg.decoder_channels = int(tj.get("decoder_channels", cfg.decoder_channels))
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="DeepLabV3+ 测试集评估脚本（各类别 IoU/Dice/P/R/F1 + 混淆矩阵）")
    parser.add_argument("--model_path", type=str, default=None, help="DeepLabV3+ best.pth 路径")
    parser.add_argument("--test_dataset_path", type=str, default=None, help="测试集根目录")
    parser.add_argument("--num_classes", type=int, default=None, help="类别数")
    parser.add_argument("--input_size", type=int, default=None, help="输入尺寸")
    parser.add_argument("--disable_postprocessing", action="store_true", help="关闭形态学后处理")
    args = parser.parse_args()

    cfg = EvalConfig()
    if args.model_path:
        cfg.model_path = args.model_path
    if args.test_dataset_path:
        cfg.test_dataset_path = args.test_dataset_path
    if args.num_classes is not None:
        cfg.num_classes = args.num_classes
    if args.input_size is not None:
        cfg.input_size = args.input_size
    if args.disable_postprocessing:
        cfg.enable_postprocessing = False

    # ✅ 按你已有编码：0..16 共 17 类
    cfg.class_names = [
         "Background", 
    "Water",
    "Bareland",
    "Bambusa",
    "Rhodomyrtus",
    "Aeschynomene",
    "Pongamia",
    "Bidens",
    "Syzygium",
    "Eichhornia",
    "Colocasia",
    "Alternanthera",
    "Musa",
    "Diplazium",
    "Arundo",
    "Pueraria",
    "Phragmites",
    ]

    # 可选后处理配置（你可以按类调参）
    cfg.postprocess_config = {
        # 示例：下面这些 class_id 仅作例子，请按你实际想后处理的类改
        # "water": {"class_id": 1, "closing_iterations": 2, "opening_iterations": 0, "min_area": 500},
    }
    ensure_class_names(cfg)
    try_load_train_config(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ✅ 构建 DeepLabV3+ 模型壳子（结构必须与训练一致）
    model = smp.DeepLabV3Plus(
        encoder_name=cfg.encoder_name,
        encoder_weights=None,  # 评估时不再从 imagenet 加载；直接 load_state_dict(best.pth)
        in_channels=3,
        classes=cfg.num_classes,
        decoder_channels=cfg.decoder_channels,
        activation=None
    ).to(device)

    if not os.path.exists(cfg.model_path):
        raise FileNotFoundError(f"模型文件不存在: {cfg.model_path}")
    ckpt = torch.load(cfg.model_path, map_location=device)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    test_txt = os.path.join(cfg.test_dataset_path, cfg.split_subdir, cfg.test_list_name)
    if not os.path.exists(test_txt):
        raise FileNotFoundError(f"测试列表不存在: {test_txt}")
    ids = read_test_ids(test_txt)

    img_root = os.path.join(cfg.test_dataset_path, cfg.img_dir)
    mask_root = os.path.join(cfg.test_dataset_path, cfg.mask_dir)

    exp_name = Path(cfg.model_path).resolve().parent.name
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(cfg.output_root) / f"DeepLabV3Plus_{exp_name}_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_pred = []
    all_gt = []

    for sid in tqdm(ids, desc="评估进度"):
        img_path = None
        for ext in (".jpg", ".JPG", ".png"):
            p = os.path.join(img_root, sid + ext)
            if os.path.exists(p):
                img_path = p
                break
        mask_path = os.path.join(mask_root, sid + ".png")
        if img_path is None or not os.path.exists(mask_path):
            continue

        x = load_rgb_image(img_path, cfg.input_size)
        y = load_mask(mask_path, cfg.input_size, cfg.num_classes)

        xt = torch.from_numpy(x).unsqueeze(0).float().to(device)
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=(cfg.use_amp and device.type == "cuda")):
                logits = model(xt)  # [1,C,H,W]
                pred = torch.argmax(logits, dim=1).cpu().numpy()[0].astype(np.uint8)

        if cfg.enable_postprocessing:
            pred = postprocess_prediction(pred, cfg.postprocess_config)

        all_pred.append(pred)
        all_gt.append(y)

    if len(all_pred) == 0:
        raise RuntimeError("没有有效测试样本，请检查 test.txt / 图像 / 掩码路径。")

    pred_flat = np.concatenate([p.flatten() for p in all_pred])
    gt_flat = np.concatenate([g.flatten() for g in all_gt])

    labels = list(range(cfg.num_classes))
    cm = confusion_matrix(gt_flat, pred_flat, labels=labels)
    iou, dice = conf_to_metrics(cm)

    miou = float(np.mean(iou))
    mdice = float(np.mean(dice))
    if cfg.num_classes > 1:
        miou_fg = float(np.mean(iou[1:]))
        mdice_fg = float(np.mean(dice[1:]))
    else:
        miou_fg, mdice_fg = miou, mdice

    pixel_acc = float((pred_flat == gt_flat).mean())
    precision, recall, f1, _ = precision_recall_fscore_support(
        gt_flat, pred_flat, labels=labels, average=None, zero_division=0
    )
    overall_precision, overall_recall, overall_f1, _ = precision_recall_fscore_support(
        gt_flat, pred_flat, labels=labels, average="macro", zero_division=0
    )

    result = {
        "model_type": "DeepLabV3Plus",
        "model_path": cfg.model_path,
        "encoder_name": cfg.encoder_name,
        "encoder_weights_train": cfg.encoder_weights,
        "decoder_channels": cfg.decoder_channels,
        "test_dataset_path": cfg.test_dataset_path,
        "num_classes": cfg.num_classes,
        "input_size": cfg.input_size,
        "num_samples": len(all_pred),
        "mean_iou": miou,
        "mean_dice": mdice,
        "overall_f1": float(overall_f1),
        "pixel_accuracy": pixel_acc,
        "overall_recall": float(overall_recall),
        "overall_precision": float(overall_precision),
        "mean_iou_fg": miou_fg,
        "mean_dice_fg": mdice_fg,
        "enable_postprocessing": cfg.enable_postprocessing,
        "class_names": cfg.class_names,
        "class_iou": [float(x) for x in iou],
        "class_dice": [float(x) for x in dice],
        "class_precision": [float(x) for x in precision],
        "class_recall": [float(x) for x in recall],
        "class_f1": [float(x) for x in f1],
        "confusion_matrix": cm.tolist(),
        "time": datetime.datetime.now().isoformat(),
    }

    with open(out_dir / "deeplabv3plus_test_evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # 混淆矩阵图
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, cmap="Blues", xticklabels=cfg.class_names, yticklabels=cfg.class_names)
    plt.title("DeepLabV3+ 测试集混淆矩阵")
    plt.xlabel("预测类别")
    plt.ylabel("真实类别")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=250, bbox_inches="tight")
    plt.close()

    # Markdown 报告
    md_lines = [
        "# DeepLabV3+ 测试集评估报告",
        "",
        f"- 模型: `{cfg.model_path}`",
        f"- Encoder: `{cfg.encoder_name}` (train pretrain: {cfg.encoder_weights})",
        f"- 测试集: `{cfg.test_dataset_path}`",
        f"- 样本数: {len(all_pred)}",
        f"- mIoU(%): {miou * 100:.2f}",
        f"- mDice(%): {mdice * 100:.2f}",
        f"- F1(%): {overall_f1 * 100:.2f}",
        f"- PA(%): {pixel_acc * 100:.2f}",
        f"- Rec(%): {overall_recall * 100:.2f}",
        f"- Pre(%): {overall_precision * 100:.2f}",
        f"- mIoU(FG): {miou_fg:.4f}",
        f"- mDice(FG): {mdice_fg:.4f}",
        f"- 后处理: {'开启' if cfg.enable_postprocessing else '关闭'}",
        "",
        "## 各类别指标",
        "",
        "| 类别 | IoU | Dice | Precision | Recall | F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for i, name in enumerate(cfg.class_names):
        md_lines.append(
            f"| {name} | {iou[i]:.4f} | {dice[i]:.4f} | {precision[i]:.4f} | {recall[i]:.4f} | {f1[i]:.4f} |"
        )
    with open(out_dir / "DEEPLABV3PLUS_TEST_EVALUATION_REPORT.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print("评估完成")
    print(f"输出目录: {out_dir}")
    print(f"mIoU(%): {miou * 100:.2f}")
    print(f"mDice(%): {mdice * 100:.2f}")
    print(f"F1(%): {overall_f1 * 100:.2f}")
    print(f"PA(%): {pixel_acc * 100:.2f}")
    print(f"Rec(%): {overall_recall * 100:.2f}")
    print(f"Pre(%): {overall_precision * 100:.2f}")
    print(f"mIoU(FG): {miou_fg:.4f}")
    print(f"mDice(FG): {mdice_fg:.4f}")


if __name__ == "__main__":
    main()