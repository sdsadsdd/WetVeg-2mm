import os
import sys
import random
import json
import datetime
from dataclasses import dataclass
from typing import List

import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    import torchvision.models as tv_models
except Exception:
    tv_models = None
try:
    from torchvision.models import ResNet18_Weights
except Exception:
    ResNet18_Weights = None


# ----------------------------
# 0) 配置：保持与你给的 UNet 脚本一致
# ----------------------------
@dataclass
class Config:
    data_root: str = r"D:\segformer-1.8\segformer-1.8\数据集\全分割16类V4.7\划分后数据集-全分割16类V4.7"

    img_dir: str = "JPEGImages"
    mask_dir: str = "SegmentationClass"

    split_dir: str = r"ImageSets\Segmentation"
    train_list: str = "train.txt"
    val_list: str = "val.txt"

    num_classes: int = 17
    ignore_index: int = 255
    img_size: int = 1024

    batch_size: int = 2
    num_workers: int = 0
    epochs: int = 200
    lr: float = 1e-4
    weight_decay: float = 1e-4
    seed: int = 42
    amp: bool = True

    base_log_dir: str = r"D:\segformer-1.8\logs-attention-unet"
    out_dir: str = ""
    save_best_metric: str = "miou"  # or "mdice"
    save_period: int = 5

    resume: bool = False
    resume_path: str = ""

    # ✅ 预训练模块：保持你原来的开关语义
    use_pretrained_backbone: bool = False
    pretrained: bool = False  # 仅当 use_pretrained_backbone=True 时生效（ImageNet）


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_id_list(txt_path: str) -> List[str]:
    with open(txt_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


class MaskSegDataset(Dataset):
    def __init__(self, ids: List[str], img_dir: str, mask_dir: str,
                 img_size: int = 1024, augment: bool = True):
        self.ids = ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.augment = augment

    def __len__(self):
        return len(self.ids)

    def _load_img(self, base: str) -> np.ndarray:
        for ext in [".jpg", ".JPG", ".png"]:
            p = os.path.join(self.img_dir, base + ext)
            if os.path.exists(p):
                return np.array(Image.open(p).convert("RGB"))
        raise FileNotFoundError(f"Image not found: {base} in {self.img_dir}")

    def _load_mask(self, base: str) -> np.ndarray:
        p = os.path.join(self.mask_dir, base + ".png")
        if not os.path.exists(p):
            raise FileNotFoundError(f"Mask not found: {base}.png in {self.mask_dir}")
        m = np.array(Image.open(p))
        if m.ndim == 3:
            m = m[:, :, 0]
        return m.astype(np.int64)

    def _augment(self, img, mask):
        if random.random() < 0.5:
            img = np.flip(img, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()
        if random.random() < 0.2:
            img = np.flip(img, axis=0).copy()
            mask = np.flip(mask, axis=0).copy()
        return img, mask

    def __getitem__(self, idx):
        base = self.ids[idx]
        img = self._load_img(base)
        mask = self._load_mask(base)

        if img.shape[0] != self.img_size or img.shape[1] != self.img_size:
            img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        if mask.shape[0] != self.img_size or mask.shape[1] != self.img_size:
            mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        if self.augment:
            img, mask = self._augment(img, mask)

        img = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        mask = torch.from_numpy(mask).long()
        return img, mask


# ----------------------------
# 1) 模型：Attention U-Net（ResNet18 预训练编码器）
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
        # 兜底对齐（极少出现1像素差）
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        skip = self.att(x, skip)
        x = self.conv(torch.cat([x, skip], dim=1))
        return x


class ResNet18AttentionUNet(nn.Module):
    """
    ResNet18 编码器（可选 ImageNet 预训练）+ Attention U-Net 解码器。
    输出与输入同分辨率：B x num_classes x H x W
    """
    def __init__(self, num_classes: int = 17, pretrained: bool = True):
        super().__init__()
        if tv_models is None:
            raise RuntimeError("需要 torchvision 才能使用 ResNet18 预训练编码器，请 pip install torchvision")

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

        # bottleneck refine
        self.center = DoubleConv(512, 512)

        # decoder (skip: x3/x2/x1/x0)
        self.up4 = UpAttBlock(512, 256, 256)  # H/16
        self.up3 = UpAttBlock(256, 128, 128)  # H/8
        self.up2 = UpAttBlock(128, 64, 64)    # H/4
        self.up1 = UpAttBlock(64, 64, 64)     # H/2 (skip来自stem)

        # 到原图分辨率 H
        self.up0 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.head = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        x0 = self.stem(x)              # 1/2, 64
        x1 = self.layer1(self.maxpool(x0))  # 1/4, 64
        x2 = self.layer2(x1)                # 1/8, 128
        x3 = self.layer3(x2)                # 1/16, 256
        x4 = self.layer4(x3)                # 1/32, 512

        c = self.center(x4)
        d4 = self.up4(c, x3)
        d3 = self.up3(d4, x2)
        d2 = self.up2(d3, x1)
        d1 = self.up1(d2, x0)
        d0 = self.up0(d1)
        return self.head(d0)


# （可选）经典纯卷积 Attention U-Net（无预训练）
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


def build_model(cfg, device: torch.device):
    if cfg.use_pretrained_backbone:
        m = ResNet18AttentionUNet(num_classes=cfg.num_classes, pretrained=cfg.pretrained).to(device)
        print(f"✅ 模型: ResNet18-AttentionUNet，ImageNet预训练={'是' if cfg.pretrained else '否'}")
    else:
        m = AttentionUNetPlain(num_classes=cfg.num_classes).to(device)
        print("✅ 模型: 纯卷积 Attention U-Net（随机初始化，无 ImageNet 预训练）")
    return m


# ----------------------------
# 2) Loss / Metrics（与你原脚本一致）
# ----------------------------
def soft_dice_loss(logits, target, num_classes, ignore_index=255, eps=1e-6):
    probs = torch.softmax(logits, dim=1)
    valid = (target != ignore_index).unsqueeze(1)
    t = target.clone()
    t[t == ignore_index] = 0
    onehot = F.one_hot(t, num_classes=num_classes).permute(0, 3, 1, 2).float()
    probs = probs * valid
    onehot = onehot * valid
    inter = (probs * onehot).sum(dim=(0, 2, 3))
    union = (probs + onehot).sum(dim=(0, 2, 3))
    dice = (2 * inter + eps) / (union + eps)
    return 1 - dice.mean()


@torch.no_grad()
def compute_confusion(pred, target, num_classes, ignore_index=255):
    pred = pred.view(-1)
    target = target.view(-1)
    m = (target != ignore_index)
    pred = pred[m]
    target = target[m]
    k = (target * num_classes + pred).long()
    conf = torch.bincount(k, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    return conf


@torch.no_grad()
def conf_to_miou_mdice(conf, eps=1e-6):
    tp = conf.diag().float()
    fp = conf.sum(0).float() - tp
    fn = conf.sum(1).float() - tp
    iou = tp / (tp + fp + fn + eps)
    dice = (2 * tp) / (2 * tp + fp + fn + eps)
    return iou.mean().item(), dice.mean().item()


def save_history(history, out_dir):
    history_path = os.path.join(out_dir, "train_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    epochs = [x["epoch"] for x in history]
    train_loss = [x["train_loss"] for x in history]
    val_loss = [x["val_loss"] for x in history]
    miou = [x["miou"] for x in history]
    mdice = [x["mdice"] for x in history]

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label="train_loss")
    plt.plot(epochs, val_loss, label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, miou, label="mIoU")
    plt.plot(epochs, mdice, label="mDice")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Validation Metrics")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "train_curves.png"), dpi=200)
    plt.close()


def make_exp_dir(base_log_dir: str):
    os.makedirs(base_log_dir, exist_ok=True)
    existing = []
    for item in os.listdir(base_log_dir):
        full = os.path.join(base_log_dir, item)
        if item.startswith("exp") and os.path.isdir(full):
            try:
                existing.append(int(item[3:]))
            except Exception:
                pass
    next_num = (max(existing) + 1) if existing else 1
    return os.path.join(base_log_dir, f"exp{next_num}")


def main():
    cfg = Config()
    try:
        cv2.setNumThreads(0)
    except Exception:
        pass

    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pin = device == "cuda"

    if not cfg.out_dir:
        if cfg.resume and cfg.resume_path:
            cfg.out_dir = os.path.dirname(cfg.resume_path)
        else:
            cfg.out_dir = make_exp_dir(cfg.base_log_dir)
    os.makedirs(cfg.out_dir, exist_ok=True)

    train_config = {
        "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_root": cfg.data_root,
        "img_dir": cfg.img_dir,
        "mask_dir": cfg.mask_dir,
        "split_dir": cfg.split_dir,
        "train_list": cfg.train_list,
        "val_list": cfg.val_list,
        "num_classes": cfg.num_classes,
        "ignore_index": cfg.ignore_index,
        "img_size": cfg.img_size,
        "batch_size": cfg.batch_size,
        "num_workers": cfg.num_workers,
        "epochs": cfg.epochs,
        "lr": cfg.lr,
        "weight_decay": cfg.weight_decay,
        "seed": cfg.seed,
        "amp": cfg.amp,
        "save_best_metric": cfg.save_best_metric,
        "save_period": cfg.save_period,
        "resume": cfg.resume,
        "resume_path": cfg.resume_path,
        "use_pretrained_backbone": cfg.use_pretrained_backbone,
        "pretrained": cfg.pretrained,
        "device": device,
        "out_dir": cfg.out_dir,
    }
    with open(os.path.join(cfg.out_dir, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump(train_config, f, ensure_ascii=False, indent=2)

    train_txt = os.path.join(cfg.data_root, cfg.split_dir, cfg.train_list)
    val_txt = os.path.join(cfg.data_root, cfg.split_dir, cfg.val_list)
    train_ids = read_id_list(train_txt)
    val_ids = read_id_list(val_txt)

    print("\n" + "=" * 80)
    print("Attention U-Net 训练开始（结构对齐你给的 UNet 脚本）")
    print("=" * 80)
    print(f"训练集: {len(train_ids)}  验证集: {len(val_ids)}")
    print(f"类别数: {cfg.num_classes}  输入尺寸: {cfg.img_size}")
    print(f"输出目录: {cfg.out_dir}")
    print("=" * 80)

    img_dir = os.path.join(cfg.data_root, cfg.img_dir)
    mask_dir = os.path.join(cfg.data_root, cfg.mask_dir)

    train_ds = MaskSegDataset(train_ids, img_dir, mask_dir, cfg.img_size, augment=True)
    val_ds = MaskSegDataset(val_ids, img_dir, mask_dir, cfg.img_size, augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=pin, drop_last=True,
        persistent_workers=cfg.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=pin,
        persistent_workers=cfg.num_workers > 0,
    )

    model = build_model(cfg, device)
    ce = nn.CrossEntropyLoss(ignore_index=cfg.ignore_index)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)

    best = -1e9
    best_path = os.path.join(cfg.out_dir, "best.pth")
    last_path = os.path.join(cfg.out_dir, "last.pth")
    history = []
    start_epoch = 1

    if cfg.resume:
        ckpt_path = cfg.resume_path if cfg.resume_path else last_path
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device)
            if isinstance(ckpt, dict) and "model" in ckpt:
                model.load_state_dict(ckpt["model"], strict=True)
                if "optimizer" in ckpt and isinstance(ckpt["optimizer"], dict):
                    opt.load_state_dict(ckpt["optimizer"])
                if cfg.amp and "scaler" in ckpt and isinstance(ckpt["scaler"], dict):
                    try:
                        scaler.load_state_dict(ckpt["scaler"])
                    except Exception:
                        pass
                history = ckpt.get("history", [])
                start_epoch = int(ckpt.get("epoch", 0)) + 1
                if history:
                    metric_key = "miou" if cfg.save_best_metric == "miou" else "mdice"
                    best = max([float(h.get(metric_key, -1e9)) for h in history], default=-1e9)
                else:
                    best = float(ckpt.get(cfg.save_best_metric, -1e9))
                print(f"[RESUME] 从断点恢复: {ckpt_path}，下一轮从 Epoch {start_epoch} 开始")

    for epoch in range(start_epoch, cfg.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs} [train]")
        train_loss_sum = 0.0
        train_steps = 0

        for imgs, masks in pbar:
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=cfg.amp):
                logits = model(imgs)
                loss = ce(logits, masks) + soft_dice_loss(logits, masks, cfg.num_classes, cfg.ignore_index)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            pbar.set_postfix(loss=float(loss.item()))
            train_loss_sum += float(loss.item())
            train_steps += 1

        model.eval()
        conf = torch.zeros((cfg.num_classes, cfg.num_classes), dtype=torch.long, device=device)
        vloss = 0.0
        with torch.no_grad():
            for imgs, masks in tqdm(val_loader, desc=f"Epoch {epoch}/{cfg.epochs} [val]"):
                imgs = imgs.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                logits = model(imgs)
                vloss += (ce(logits, masks) + soft_dice_loss(logits, masks, cfg.num_classes, cfg.ignore_index)).item()
                pred = torch.argmax(logits, dim=1)
                conf += compute_confusion(pred, masks, cfg.num_classes, cfg.ignore_index).to(device)

        miou, mdice = conf_to_miou_mdice(conf)
        train_loss = train_loss_sum / max(1, train_steps)
        vloss /= max(1, len(val_loader))
        score = miou if cfg.save_best_metric == "miou" else mdice

        history.append({
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(vloss),
            "miou": float(miou),
            "mdice": float(mdice),
            "lr": float(opt.param_groups[0]["lr"]),
        })
        save_history(history, cfg.out_dir)

        print(f"[Epoch {epoch}] train_loss={train_loss:.4f}  val_loss={vloss:.4f}  mIoU={miou:.4f}  mDice={mdice:.4f}")

        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "scaler": scaler.state_dict() if cfg.amp else {},
                "miou": miou,
                "mdice": mdice,
                "history": history,
            },
            last_path
        )

        if (epoch % cfg.save_period) == 0:
            ep_path = os.path.join(cfg.out_dir, f"ep{epoch:03d}.pth")
            torch.save({"epoch": epoch, "model": model.state_dict(), "miou": miou, "mdice": mdice}, ep_path)

        if score > best:
            best = score
            torch.save({"epoch": epoch, "model": model.state_dict(), "miou": miou, "mdice": mdice}, best_path)
            print(f"[SAVE] best -> {best_path} (score={best:.4f})")

    print(f"Done. 历史已保存: {os.path.join(cfg.out_dir, 'train_history.json')}")
    print(f"曲线已保存: {os.path.join(cfg.out_dir, 'train_curves.png')}")
    print(f"最佳模型: {best_path}")
    print(f"最新模型: {last_path}")


if __name__ == "__main__":
    main()