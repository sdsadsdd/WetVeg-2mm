import os
import sys
import random
import json
import datetime
from dataclasses import dataclass
from typing import List, Tuple

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
# 0) 配置：你只需改 data_root 和图像目录名
# ----------------------------
@dataclass
class Config:
    data_root: str = r"D:\segformer-1.8\segformer-1.8\数据集\全分割16类V4.7\划分后数据集-全分割16类V4.7"  # 改成你的根目录

    # 原图目录（二选一：JPEGImages 或 png）
    img_dir: str = "JPEGImages"          # 如果你的原图是jpg
    # img_dir: str = "png"               # 如果你的原图是png

    # mask目录（你说你已经有了）
    mask_dir: str = "SegmentationClass"  # 你的掩码png

    # split txt
    split_dir: str = r"ImageSets\Segmentation"
    train_list: str = "train.txt"
    val_list: str = "val.txt"

    # 你这套 classes 是 0..16 => 17 类
    num_classes: int = 17

    # 如果你的mask里没有ignore，就保持255但不会出现
    ignore_index: int = 255

    # tile size
    img_size: int = 1024

    # train
    batch_size: int = 2
    # Windows + spawn 下 num_workers>0 常与 OpenCV/PIL 冲突，子进程崩了只报
    # "DataLoader worker exited unexpectedly"。默认 0 最稳；Linux 可改为 2~4。
    num_workers: int = 0 
    epochs: int = 200
    lr: float = 1e-4
    weight_decay: float = 1e-4
    seed: int = 42
    amp: bool = True

    # 实验输出根目录：会自动创建 exp1/exp2...
    base_log_dir: str = r"D:\segformer-1.8\logs-u-net"
    out_dir: str = ""  # 留空则自动使用 base_log_dir/expN
    save_best_metric: str = "miou"  # or "mdice"
    save_period: int = 5
    # 断点续训配置
    resume: bool = False
    resume_path: str = ""  # 为空时默认使用 out_dir/last.pth

    # 预训练：True 时使用 torchvision ResNet18（ImageNet1K）作编码器 + U-Net 解码器
    # False 时为经典纯卷积 U-Net（随机初始化，无 ImageNet）
    use_pretrained_backbone: bool = True
    pretrained: bool = True  # 仅当 use_pretrained_backbone=True 时生效；加载 ImageNet 权重


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
        # 尝试 jpg/png
        for ext in [".jpg", ".JPG", ".png"]:
            p = os.path.join(self.img_dir, base + ext)
            if os.path.exists(p):
                return np.array(Image.open(p).convert("RGB"))
        raise FileNotFoundError(f"Image not found: {base} in {self.img_dir}")

    def _load_mask(self, base: str) -> np.ndarray:
        p = os.path.join(self.mask_dir, base + ".png")
        if not os.path.exists(p):
            raise FileNotFoundError(f"Mask not found: {base}.png in {self.mask_dir}")
        # mask是单通道，像素值为类别ID
        m = np.array(Image.open(p))
        if m.ndim == 3:
            # 有些mask会存成RGB，这里兜底：取单通道
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

        # resize到固定size（如果本来就是1024，可省略）
        if img.shape[0] != self.img_size or img.shape[1] != self.img_size:
            img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        if mask.shape[0] != self.img_size or mask.shape[1] != self.img_size:
            mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        if self.augment:
            img, mask = self._augment(img, mask)

        img = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        mask = torch.from_numpy(mask).long()
        return img, mask


# ----- U-Net -----
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
        self.enc2 = DoubleConv(base, base*2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(base*2, base*4)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = DoubleConv(base*4, base*8)
        self.pool4 = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(base*8, base*16)

        self.up4 = nn.ConvTranspose2d(base*16, base*8, 2, stride=2)
        self.dec4 = DoubleConv(base*16, base*8)
        self.up3 = nn.ConvTranspose2d(base*8, base*4, 2, stride=2)
        self.dec3 = DoubleConv(base*8, base*4)
        self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.dec2 = DoubleConv(base*4, base*2)
        self.up1 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.dec1 = DoubleConv(base*2, base)
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
    """
    ResNet18 编码器（可选 ImageNet 预训练）+ U-Net 式解码器。
    输入 3xHxW，输出 num_classes x H x W（与输入同分辨率）。
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
        # stem 输出为 H/2；再上一倍到与原图同分辨率（无对应 skip，仅上采样）
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


def build_model(cfg, device: torch.device):
    if cfg.use_pretrained_backbone:
        m = ResNet18UNet(num_classes=cfg.num_classes, pretrained=cfg.pretrained).to(device)
        print(
            f"✅ 模型: ResNet18-UNet，编码器预训练(ImageNet)={'是' if cfg.pretrained else '否'}"
        )
    else:
        m = UNet(num_classes=cfg.num_classes).to(device)
        print("✅ 模型: 经典 U-Net（随机初始化，无 ImageNet 预训练）")
    return m


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
    conf = torch.bincount(k, minlength=num_classes*num_classes).reshape(num_classes, num_classes)
    return conf

@torch.no_grad()
def conf_to_miou_mdice(conf, eps=1e-6):
    tp = conf.diag().float()
    fp = conf.sum(0).float() - tp
    fn = conf.sum(1).float() - tp
    iou = tp / (tp + fp + fn + eps)
    dice = (2*tp) / (2*tp + fp + fn + eps)
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
    # 避免 OpenCV 多线程与 DataLoader 子进程（Windows）死锁/崩溃
    try:
        cv2.setNumThreads(0)
    except Exception:
        pass
    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
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
    print("U-Net 训练开始")
    print("=" * 80)
    print(f"训练集: {len(train_ids)}  验证集: {len(val_ids)}")
    print(f"类别数: {cfg.num_classes}  输入尺寸: {cfg.img_size}")
    print(f"输出目录: {cfg.out_dir}")
    print("=" * 80)

    img_dir = os.path.join(cfg.data_root, cfg.img_dir)
    mask_dir = os.path.join(cfg.data_root, cfg.mask_dir)

    train_ds = MaskSegDataset(train_ids, img_dir, mask_dir, cfg.img_size, augment=True)
    val_ds = MaskSegDataset(val_ids, img_dir, mask_dir, cfg.img_size, augment=False)

    pin = device == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=pin,
        drop_last=True,
        persistent_workers=cfg.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin,
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

    # 断点续训：恢复模型、优化器、AMP状态与历史
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
                # 恢复 best 参考值（与 save_best_metric 一致）
                if history:
                    metric_key = "miou" if cfg.save_best_metric == "miou" else "mdice"
                    best = max([float(h.get(metric_key, -1e9)) for h in history], default=-1e9)
                else:
                    best = float(ckpt.get(cfg.save_best_metric, -1e9))
                print(f"[RESUME] 从断点恢复: {ckpt_path}，下一轮从 Epoch {start_epoch} 开始")
            else:
                print(f"[RESUME] 警告：断点文件格式不正确，已忽略: {ckpt_path}")
        else:
            print(f"[RESUME] 警告：未找到断点文件，已从头训练: {ckpt_path}")

    if start_epoch > cfg.epochs:
        print(f"已完成训练：start_epoch={start_epoch} > epochs={cfg.epochs}")
        return

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