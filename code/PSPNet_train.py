import os
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

import segmentation_models_pytorch as smp


# ----------------------------
# 0) Config（与你其它 baseline 脚本一致风格）
# ----------------------------
@dataclass
class Config:
    data_root: str = r"D:\segformer-1.8\segformer-1.8\数据集\全分割16类V4.7\划分后数据集-全分割16类V4.7"

    img_dir: str = "JPEGImages"          # 或 "png"
    mask_dir: str = "SegmentationClass"
    split_dir: str = r"ImageSets\Segmentation"
    train_list: str = "train.txt"
    val_list: str = "val.txt"

    # 你的mask编码：0..16 => 17类（含背景）
    num_classes: int = 17
    ignore_index: int = 255
    img_size: int = 1024

    batch_size: int = 2
    num_workers: int = 0              # Windows 推荐先 0
    epochs: int = 100
    lr: float = 1e-4
    weight_decay: float = 1e-4
    seed: int = 42
    amp: bool = True
    save_best_metric: str = "miou"    # or "mdice"
    save_period: int = 5

    # 输出目录
    base_log_dir: str = r"D:\segformer-1.8\logs-pspnet"
    out_dir: str = ""                 # 留空则自动 expN

    # ✅ 预训练模块（关键）
    encoder_name: str = "resnet50"    # 常用：resnet50/resnet101
    encoder_weights: str = "imagenet" # "imagenet" or None，写None就是关闭预训练

    # 断点续训
    resume: bool = False
    resume_path: str = r""   #"D:\segformer-1.8\logs-pspnet\exp1\last.pth"
    

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
# Loss / Metrics
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

    # Windows/OpenCV + DataLoader 更稳
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

    # 记录训练配置
    train_config = {
        "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": "PSPNet",
        "encoder_name": cfg.encoder_name,
        "encoder_weights": cfg.encoder_weights,
        "data_root": cfg.data_root,
        "img_dir": cfg.img_dir,
        "mask_dir": cfg.mask_dir,
        "split_dir": cfg.split_dir,
        "train_list": cfg.train_list,
        "val_list": cfg.val_list,
        "num_classes": cfg.num_classes,
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
    print("PSPNet 训练开始（支持 ImageNet 预训练）")
    print("=" * 80)
    print(f"Train={len(train_ids)}  Val={len(val_ids)}")
    print(f"Backbone={cfg.encoder_name}  Pretrain={cfg.encoder_weights}")
    print(f"num_classes={cfg.num_classes}  img_size={cfg.img_size}")
    print(f"out_dir={cfg.out_dir}")
    print("=" * 80)

    img_dir = os.path.join(cfg.data_root, cfg.img_dir)
    mask_dir = os.path.join(cfg.data_root, cfg.mask_dir)

    train_ds = MaskSegDataset(train_ids, img_dir, mask_dir, cfg.img_size, augment=True)
    val_ds = MaskSegDataset(val_ids, img_dir, mask_dir, cfg.img_size, augment=False)

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

    # ✅ PSPNet with pretrained encoder
    model = smp.PSPNet(
        encoder_name=cfg.encoder_name,
        encoder_weights=cfg.encoder_weights,  # "imagenet" or None
        in_channels=3,
        classes=cfg.num_classes,
        activation=None,
    ).to(device)

    ce = nn.CrossEntropyLoss(ignore_index=cfg.ignore_index)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)

    best = -1e9
    best_path = os.path.join(cfg.out_dir, "best.pth")
    last_path = os.path.join(cfg.out_dir, "last.pth")
    history = []
    start_epoch = 1

    # resume
    if cfg.resume:
        ckpt_path = cfg.resume_path if cfg.resume_path else last_path
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model"], strict=True)
            opt.load_state_dict(ckpt.get("optimizer", opt.state_dict()))
            if cfg.amp and "scaler" in ckpt:
                try:
                    scaler.load_state_dict(ckpt["scaler"])
                except Exception:
                    pass
            history = ckpt.get("history", [])
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            if history:
                key = "miou" if cfg.save_best_metric == "miou" else "mdice"
                best = max([float(h.get(key, -1e9)) for h in history], default=-1e9)
            print(f"[RESUME] {ckpt_path}, next epoch={start_epoch}")
        else:
            print(f"[RESUME] Not found: {ckpt_path}, start from scratch.")

    for epoch in range(start_epoch, cfg.epochs + 1):
        # train
        model.train()
        train_loss_sum = 0.0
        steps = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs} [train]")
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

            train_loss_sum += float(loss.item())
            steps += 1
            pbar.set_postfix(loss=float(loss.item()))

        train_loss = train_loss_sum / max(1, steps)

        # val
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

        vloss /= max(1, len(val_loader))
        miou, mdice = conf_to_miou_mdice(conf)
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

        # last
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scaler": scaler.state_dict() if cfg.amp else {},
            "miou": miou,
            "mdice": mdice,
            "history": history,
            "config": train_config
        }, last_path)

        # periodic
        if (epoch % cfg.save_period) == 0:
            ep_path = os.path.join(cfg.out_dir, f"ep{epoch:03d}.pth")
            torch.save({"epoch": epoch, "model": model.state_dict(), "miou": miou, "mdice": mdice}, ep_path)

        # best
        if score > best:
            best = score
            torch.save({"epoch": epoch, "model": model.state_dict(), "miou": miou, "mdice": mdice}, best_path)
            print(f"[SAVE] best -> {best_path} (score={best:.4f})")

    print("Done.")
    print(f"best: {best_path}")
    print(f"last: {last_path}")
    print(f"curves: {os.path.join(cfg.out_dir, 'train_curves.png')}")


if __name__ == "__main__":
    main()