import os
import json
import argparse
import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

# 画混淆矩阵（与原脚本保持一致）
import matplotlib.pyplot as plt
import seaborn as sns

# 可选后处理（与原脚本一致）
from skimage.morphology import binary_closing, binary_opening, remove_small_objects

try:
    import torchvision.models as tv_models
    from torchvision.models import ResNet18_Weights
except Exception:
    tv_models = None
    ResNet18_Weights = None


# ----------------------------
# 0) Eval Config（保持你U-Net评估脚本风格）
# ----------------------------
@dataclass
class EvalConfig:
    model_path: str = r"D:\segformer-1.8\logs-attention-unet\exp2\best.pth"
    test_dataset_path: str = r"D:\segformer-1.8\segformer-1.8\数据集\全分割16类V4.7\划分后数据集-全分割16类V4.7"
    num_classes: int = 17
    input_size: int = 1024
    ignore_index: int = 255

    # 是否启用AMP推理
    use_amp: bool = True

    # 是否后处理
    enable_postprocessing: bool = True
    postprocess_config: dict = None

    # 类别名
    class_names: List[str] = None

    # 是否使用ResNet编码器版Attention U-Net（对应“预训练模块”的那种）
    use_pretrained_backbone: bool = False


def ensure_class_names(cfg: EvalConfig):
    if cfg.class_names is None or len(cfg.class_names) != cfg.num_classes:
        raise ValueError(
            f"class_names 长度不匹配：len={len(cfg.class_names) if cfg.class_names else None}, "
            f"num_classes={cfg.num_classes}"
        )


# ----------------------------
# 1) I/O：读取图像/掩码（与你的U-Net评估脚本一致）
# ----------------------------
def find_image_path(img_dir: Path, stem: str) -> Path:
    for ext in [".jpg", ".JPG", ".png"]:
        p = img_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def load_image(path: Path, size: int) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    arr = np.array(im, dtype=np.uint8)
    arr = cv2.resize(arr, (size, size), interpolation=cv2.INTER_LINEAR)
    # CHW, float32 [0,1]
    x = arr.transpose(2, 0, 1).astype(np.float32) / 255.0
    return x


def load_mask(path: Path, size: int, num_classes: int) -> np.ndarray:
    m = Image.open(path)
    if m.mode == "P":
        arr = np.array(m, dtype=np.uint8)
    else:
        arr = np.array(m.convert("L"), dtype=np.uint8)
    arr = cv2.resize(arr, (size, size), interpolation=cv2.INTER_NEAREST)
    # 超范围兜底
    arr = np.where(arr >= num_classes, 0, arr).astype(np.uint8)
    return arr


def read_split_ids(split_txt: Path) -> List[str]:
    with open(split_txt, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


# ----------------------------
# 2) Metrics（与你原脚本一致）
# ----------------------------
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


# ----------------------------
# 3) Attention U-Net 模型定义
#    - 纯卷积版
#    - ResNet18 编码器版（可对应预训练）
# ----------------------------
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


class AttentionGate(nn.Module):
    """g: decoder feature, x: encoder skip"""
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        psi = self.relu(self.W_g(g) + self.W_x(x))
        psi = self.psi(psi)
        return x * psi


class UpAttBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.att = AttentionGate(F_g=out_ch, F_l=skip_ch, F_int=max(out_ch // 2, 16))
        self.conv = DoubleConv(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        skip = self.att(x, skip)
        x = self.conv(torch.cat([x, skip], dim=1))
        return x


class AttentionUNetPlain(nn.Module):
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
        self.att4 = AttentionGate(base * 8, base * 8, base * 4)
        self.dec4 = DoubleConv(base * 16, base * 8)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.att3 = AttentionGate(base * 4, base * 4, base * 2)
        self.dec3 = DoubleConv(base * 8, base * 4)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.att2 = AttentionGate(base * 2, base * 2, base)
        self.dec2 = DoubleConv(base * 4, base * 2)

        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.att1 = AttentionGate(base, base, max(base // 2, 16))
        self.dec1 = DoubleConv(base * 2, base)

        self.head = nn.Conv2d(base, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        b = self.bottleneck(self.pool4(e4))

        d4 = self.up4(b)
        e4a = self.att4(d4, e4)
        d4 = self.dec4(torch.cat([d4, e4a], dim=1))

        d3 = self.up3(d4)
        e3a = self.att3(d3, e3)
        d3 = self.dec3(torch.cat([d3, e3a], dim=1))

        d2 = self.up2(d3)
        e2a = self.att2(d2, e2)
        d2 = self.dec2(torch.cat([d2, e2a], dim=1))

        d1 = self.up1(d2)
        e1a = self.att1(d1, e1)
        d1 = self.dec1(torch.cat([d1, e1a], dim=1))

        return self.head(d1)


class ResNet18AttentionUNet(nn.Module):
    """ResNet18 编码器版 Attention U-Net（与“预训练模块”对应）"""
    def __init__(self, num_classes=17, pretrained=True):
        super().__init__()
        if tv_models is None:
            raise RuntimeError("需要 torchvision 才能使用 ResNet18 编码器，请 pip install torchvision")

        if ResNet18_Weights is not None:
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = tv_models.resnet18(weights=weights)
        else:
            backbone = tv_models.resnet18(pretrained=pretrained)

        # encoder
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)  # H/2, 64
        self.maxpool = backbone.maxpool                                        # H/4
        self.layer1 = backbone.layer1  # H/4, 64
        self.layer2 = backbone.layer2  # H/8, 128
        self.layer3 = backbone.layer3  # H/16, 256
        self.layer4 = backbone.layer4  # H/32, 512

        self.center = DoubleConv(512, 512)

        self.up4 = UpAttBlock(512, 256, 256)   # -> H/16
        self.up3 = UpAttBlock(256, 128, 128)   # -> H/8
        self.up2 = UpAttBlock(128, 64, 64)     # -> H/4
        self.up1 = UpAttBlock(64, 64, 64)      # -> H/2 (skip=stem)

        self.up0 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)  # -> H
        self.head = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        x0 = self.stem(x)                 # H/2, 64
        x = self.maxpool(x0)              # H/4
        x1 = self.layer1(x)               # H/4, 64
        x2 = self.layer2(x1)              # H/8, 128
        x3 = self.layer3(x2)              # H/16, 256
        x4 = self.layer4(x3)              # H/32, 512

        c = self.center(x4)
        d4 = self.up4(c, x3)
        d3 = self.up3(d4, x2)
        d2 = self.up2(d3, x1)
        d1 = self.up1(d2, x0)
        d0 = self.up0(d1)
        return self.head(d0)


def build_model(cfg: EvalConfig, device: torch.device):
    if cfg.use_pretrained_backbone:
        model = ResNet18AttentionUNet(num_classes=cfg.num_classes, pretrained=False).to(device)
        print("评估使用模型: ResNet18-AttentionUNet（与训练 use_pretrained_backbone=True 对齐）")
    else:
        model = AttentionUNetPlain(in_ch=3, num_classes=cfg.num_classes, base=64).to(device)
        print("评估使用模型: Attention U-Net（纯卷积版）")
    return model


# ----------------------------
# 4) Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Attention U-Net 测试集评估脚本")
    parser.add_argument("--model_path", type=str, default=None, help="best.pth 路径")
    parser.add_argument("--test_dataset_path", type=str, default=None, help="测试集根目录")
    parser.add_argument("--num_classes", type=int, default=None, help="类别数")
    parser.add_argument("--input_size", type=int, default=None, help="输入尺寸")
    parser.add_argument("--disable_postprocessing", action="store_true", help="关闭形态学后处理")
    parser.add_argument(
        "--use_resnet_backbone",
        action="store_true",
        help="使用 ResNet18 编码器版 Attention U-Net（与训练 use_pretrained_backbone=True 一致）",
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

    # 类别名（与你数据集一致）
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
    ensure_class_names(cfg)

    # 后处理配置：你原脚本里是给某些类写的，这里保留接口（默认不影响）
    cfg.postprocess_config = cfg.postprocess_config or {}

    # 自动读取 train_config.json 推断是否使用“预训练编码器版本”
    train_cfg_path = Path(cfg.model_path).resolve().parent / "train_config.json"
    if train_cfg_path.exists():
        try:
            with open(train_cfg_path, "r", encoding="utf-8") as f:
                tj = json.load(f)
            cfg.use_pretrained_backbone = bool(tj.get("use_pretrained_backbone", False))
        except Exception:
            pass

    # 命令行强制优先
    if args.use_resnet_backbone:
        cfg.use_pretrained_backbone = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, device)

    ckpt = torch.load(cfg.model_path, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"], strict=True)
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"], strict=True)
    else:
        # 兼容直接保存 state_dict 的情况
        model.load_state_dict(ckpt, strict=True)

    model.eval()

    root = Path(cfg.test_dataset_path)
    img_dir = root / "JPEGImages"
    mask_dir = root / "SegmentationClass"
    split_txt = root / "ImageSets" / "Segmentation" / "test.txt"
    if not split_txt.exists():
        raise FileNotFoundError(f"找不到 test.txt: {split_txt}")

    ids = read_split_ids(split_txt)

    out_dir = Path(cfg.model_path).resolve().parent / "test_eval_attention_unet"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_pred = []
    all_gt = []

    for stem in tqdm(ids, desc="Evaluating"):
        img_path = find_image_path(img_dir, stem)
        if img_path is None:
            print(f"[WARN] missing image: {stem}")
            continue
        mask_path = mask_dir / f"{stem}.png"
        if not mask_path.exists():
            print(f"[WARN] missing mask: {stem}.png")
            continue

        x = load_image(img_path, cfg.input_size)
        y = load_mask(mask_path, cfg.input_size, cfg.num_classes)

        xt = torch.from_numpy(x).unsqueeze(0).to(device)
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
    miou_fg = float(np.mean(iou[1:])) if cfg.num_classes > 1 else miou
    mdice_fg = float(np.mean(dice[1:])) if cfg.num_classes > 1 else mdice

    pixel_acc = float((pred_flat == gt_flat).mean())

    precision, recall, f1, _ = precision_recall_fscore_support(
        gt_flat, pred_flat, labels=labels, average=None, zero_division=0
    )
    overall_precision, overall_recall, overall_f1, _ = precision_recall_fscore_support(
        gt_flat, pred_flat, labels=labels, average="macro", zero_division=0
    )

    # 输出 JSON
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
        "use_pretrained_backbone": cfg.use_pretrained_backbone,
        "class_names": cfg.class_names,
        "class_iou": [float(x) for x in iou],
        "class_dice": [float(x) for x in dice],
        "class_precision": [float(x) for x in precision],
        "class_recall": [float(x) for x in recall],
        "class_f1": [float(x) for x in f1],
        "confusion_matrix": cm.tolist(),
        "time": datetime.datetime.now().isoformat(),
    }
    with open(out_dir / "attention_unet_test_evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # 混淆矩阵图
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, cmap="Blues", xticklabels=cfg.class_names, yticklabels=cfg.class_names)
    plt.title("Attention U-Net 测试集混淆矩阵")
    plt.xlabel("预测类别")
    plt.ylabel("真实类别")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=250, bbox_inches="tight")
    plt.close()

    # Markdown 报告（与你原脚本同风格）
    md_lines = [
        "# Attention U-Net 测试集评估报告",
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
        f"- ResNet18 编码器版: {'是' if cfg.use_pretrained_backbone else '否'}",
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
    with open(out_dir / "ATTENTION_UNET_TEST_EVALUATION_REPORT.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print("\n评估完成")
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