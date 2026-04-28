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

try:
    import torchvision.models as tv_models
except Exception:
    tv_models = None
try:
    from torchvision.models import ResNet18_Weights
except Exception:
    ResNet18_Weights = None


@dataclass
class EvalConfig:
    model_path: str = r"D:\segformer-1.8\logs-u-net\exp6\best.pth"
    test_dataset_path: str = r"D:\segformer-1.8\segformer-1.8\数据集\全分割16类V4.7\划分后数据集-全分割16类V4.7"
    img_dir: str = "JPEGImages"
    mask_dir: str = "SegmentationClass"
    split_subdir: str = r"ImageSets\Segmentation"
    test_list_name: str = "test.txt"
    input_size: int = 1024
    num_classes: int = 17
    ignore_index: int = 255
    use_amp: bool = True
    output_root: str = "test_evaluation_results"
    enable_postprocessing: bool = True
    postprocess_config: dict = None
    # 注意：长度需和 num_classes 一致
    class_names: List[str] = None
    # 与训练一致：ResNet18 编码器版 U-Net；默认从权重同目录 train_config.json 读取
    use_pretrained_backbone: bool = False


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_ch=3, num_classes=17, base=64):
        super().__init__()
        self.enc1 = DoubleConv(in_ch, base)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(base * 2, base * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = DoubleConv(base * 4, base * 8)
        self.pool4 = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(base * 8, base * 16)

        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.dec4 = DoubleConv(base * 16, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = DoubleConv(base * 2, base)
        self.head = nn.Conv2d(base, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        b = self.bottleneck(self.pool4(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


class ResNet18UNet(nn.Module):
    """与 7.U-net_train.py 中结构一致，便于加载 best.pth。"""

    def __init__(self, num_classes: int = 17, pretrained: bool = False):
        super().__init__()
        if tv_models is None:
            raise RuntimeError("需要 torchvision 才能加载 ResNet18-UNet 权重")
        if ResNet18_Weights is not None:
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = tv_models.resnet18(weights=weights)
        else:
            backbone = tv_models.resnet18(pretrained=pretrained)

        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)
        self.up0 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.head = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        x0 = self.stem(x)
        x = self.maxpool(x0)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        u = self.up4(x4)
        u = self.dec4(torch.cat([u, x3], dim=1))
        u = self.up3(u)
        u = self.dec3(torch.cat([u, x2], dim=1))
        u = self.up2(u)
        u = self.dec2(torch.cat([u, x1], dim=1))
        u = self.up1(u)
        u = self.dec1(torch.cat([u, x0], dim=1))
        u = self.up0(u)
        return self.head(u)


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


def main():
    parser = argparse.ArgumentParser(description="U-Net 测试集评估脚本")
    parser.add_argument("--model_path", type=str, default=None, help="U-Net best.pth 路径")
    parser.add_argument("--test_dataset_path", type=str, default=None, help="测试集根目录")
    parser.add_argument("--num_classes", type=int, default=None, help="类别数")
    parser.add_argument("--input_size", type=int, default=None, help="输入尺寸")
    parser.add_argument("--disable_postprocessing", action="store_true", help="关闭形态学后处理")
    parser.add_argument(
        "--use_resnet_backbone",
        action="store_true",
        help="使用 ResNet18 编码器版 U-Net（与训练 use_pretrained_backbone=True 一致）",
    )
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
    cfg.postprocess_config = {
        "alocasiaodora": {"class_id": 1, "closing_iterations": 3, "opening_iterations": 2, "min_area": 200},
        "eichhornia": {"class_id": 2, "closing_iterations": 2, "opening_iterations": 1, "min_area": 150},
        "cyperus": {"class_id": 3, "closing_iterations": 2, "opening_iterations": 1, "min_area": 150},
    }
    ensure_class_names(cfg)

    train_cfg_path = Path(cfg.model_path).resolve().parent / "train_config.json"
    if train_cfg_path.exists():
        with open(train_cfg_path, "r", encoding="utf-8") as f:
            tj = json.load(f)
        cfg.use_pretrained_backbone = bool(tj.get("use_pretrained_backbone", False))
    if args.use_resnet_backbone:
        cfg.use_pretrained_backbone = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cfg.use_pretrained_backbone:
        model = ResNet18UNet(num_classes=cfg.num_classes, pretrained=False).to(device)
        print("评估使用模型: ResNet18-UNet（与训练一致）")
    else:
        model = UNet(in_ch=3, num_classes=cfg.num_classes).to(device)
        print("评估使用模型: 经典 U-Net")

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
    out_dir = Path(cfg.output_root) / f"Unet_{exp_name}_{run_id}"
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
                logits = model(xt)
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
        miou_fg = miou
        mdice_fg = mdice
    pixel_acc = float((pred_flat == gt_flat).mean())
    precision, recall, f1, _ = precision_recall_fscore_support(
        gt_flat, pred_flat, labels=labels, average=None, zero_division=0
    )
    overall_precision, overall_recall, overall_f1, _ = precision_recall_fscore_support(
        gt_flat, pred_flat, labels=labels, average="macro", zero_division=0
    )

    result = {
        "model_path": cfg.model_path,
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

    with open(out_dir / "unet_test_evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, cmap="Blues", xticklabels=cfg.class_names, yticklabels=cfg.class_names)
    plt.title("U-Net 测试集混淆矩阵")
    plt.xlabel("预测类别")
    plt.ylabel("真实类别")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=250, bbox_inches="tight")
    plt.close()

    md_lines = [
        "# U-Net 测试集评估报告",
        "",
        f"- 模型: `{cfg.model_path}`",
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
    with open(out_dir / "UNET_TEST_EVALUATION_REPORT.md", "w", encoding="utf-8") as f:
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

