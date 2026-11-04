from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import functional as TF


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class UNet(nn.Module):
    def __init__(self, base: int = 32):
        super().__init__()
        self.enc1, self.enc2 = ConvBlock(3, base), ConvBlock(base, base * 2)
        self.enc3, self.enc4 = ConvBlock(base * 2, base * 4), ConvBlock(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(base * 8, base * 16)
        self.up4, self.dec4 = nn.ConvTranspose2d(base * 16, base * 8, 2, 2), ConvBlock(base * 16, base * 8)
        self.up3, self.dec3 = nn.ConvTranspose2d(base * 8, base * 4, 2, 2), ConvBlock(base * 8, base * 4)
        self.up2, self.dec2 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2), ConvBlock(base * 4, base * 2)
        self.up1, self.dec1 = nn.ConvTranspose2d(base * 2, base, 2, 2), ConvBlock(base * 2, base)
        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        a = self.enc1(x)
        b = self.enc2(self.pool(a))
        c = self.enc3(self.pool(b))
        d = self.enc4(self.pool(c))
        z = self.bottleneck(self.pool(d))
        z = self.dec4(torch.cat([self.up4(z), d], dim=1))
        z = self.dec3(torch.cat([self.up3(z), c], dim=1))
        z = self.dec2(torch.cat([self.up2(z), b], dim=1))
        z = self.dec1(torch.cat([self.up1(z), a], dim=1))
        return self.out(z)


class MaskDataset(Dataset):
    def __init__(self, image_dir: Path, mask_dir: Path, size: int):
        images = {p.stem: p for p in image_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg"}}
        masks = {p.stem.removesuffix("_segmentation"): p for p in mask_dir.rglob("*.png")}
        missing = sorted(set(images) - set(masks))
        extra = sorted(set(masks) - set(images))
        if missing or extra:
            raise ValueError(f"Image/mask mismatch: {len(missing)} images lack masks; {len(extra)} masks lack images")
        if not images:
            raise FileNotFoundError(f"No JPG images found in {image_dir}")
        self.pairs, self.size = [(images[k], masks[k]) for k in sorted(images)], size

    def __len__(self): return len(self.pairs)

    def __getitem__(self, index):
        image_path, mask_path = self.pairs[index]
        with Image.open(image_path) as im: image = im.convert("RGB")
        with Image.open(mask_path) as im: mask = im.convert("L")
        image = TF.resize(image, [self.size, self.size], interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, [self.size, self.size], interpolation=TF.InterpolationMode.NEAREST)
        x = TF.to_tensor(image)
        x = TF.normalize(x, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        y = (TF.to_tensor(mask) > 0.5).float()
        return x, y


def loss_and_dice(logits, target):
    bce = nn.functional.binary_cross_entropy_with_logits(logits, target)
    prob = torch.sigmoid(logits)
    intersection = (prob * target).sum(dim=(1, 2, 3))
    dice = ((2 * intersection + 1) / (prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + 1)).mean()
    return bce + (1 - dice), dice


def main():
    parser = argparse.ArgumentParser(description="Train a U-Net on ISIC 2018 lesion masks")
    parser.add_argument("--images", required=True)
    parser.add_argument("--masks", required=True)
    parser.add_argument("--output", default="runs/isic2018_segmenter/best_segmenter.pt")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.size % 16:
        parser.error("epochs/batch-size must be positive and --size must be divisible by 16")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")
    dataset = MaskDataset(Path(args.images), Path(args.masks), args.size)
    n_val = max(1, round(len(dataset) * 0.15))
    train_set, val_set = random_split(dataset, [len(dataset) - n_val, n_val], generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=device.type == "cuda")
    model = UNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    # Use the current AMP API on recent PyTorch; retain support for older installs.
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    best, stale = -1.0, 0
    for epoch in range(1, args.epochs + 1):
        model.train(); train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(x); loss, _ = loss_and_dice(logits, y)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            train_loss += loss.item()
        model.eval(); dice_sum = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                logits = model(x)
                _, dice = loss_and_dice(logits.float(), y)
                dice_sum += dice.item() * len(x)
        val_dice = dice_sum / len(val_set)
        print(f"epoch {epoch}/{args.epochs}: train_loss={train_loss / len(train_loader):.4f}, val_dice={val_dice:.4f}")
        if val_dice > best:
            best, stale = val_dice, 0
            torch.save({"model_state": model.state_dict(), "size": args.size, "val_dice": best}, output)
        else:
            stale += 1
            if stale >= 8:
                print("Early stopping: validation Dice did not improve."); break
    print(f"Best validation Dice: {best:.4f}; checkpoint: {output}")


if __name__ == "__main__": main()
