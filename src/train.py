import os
import argparse
import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from oxford_pet import OxfordPetDataset
from models.unet import UNet
from models.resnet34_unet import ResNet34_UNet
from evaluate import evaluate
from utils import FocalTverskyLoss, dice_loss_from_logits, EMA


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(
    Epochs=60,
    Batch_size=24,
    Learning_rate=1e-4,
    model_type="ResNet34_UNet",
    disable_amp=False,
):
    set_seed(42)

    project_root = os.path.abspath(os.getcwd())
    data_dir = os.path.join(project_root, "dataset")

    if model_type == "UNet":
        img_size, mask_size, accumulation_steps = 572, 388, 4
    else:
        img_size, mask_size, accumulation_steps = 256, 256, 1

    train_dataset = OxfordPetDataset(
        data_dir=data_dir,
        split="train",
        image_size=img_size,
        mask_size=mask_size,
    )
    val_dataset = OxfordPetDataset(
        data_dir=data_dir,
        split="val",
        image_size=img_size,
        mask_size=mask_size,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    if use_cuda:
        torch.backends.cudnn.benchmark = True

    num_workers = min(4, os.cpu_count()) if os.cpu_count() else 2
    train_loader = DataLoader(
        train_dataset,
        batch_size=Batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_cuda,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    model = UNet().to(device) if model_type == "UNet" else ResNet34_UNet().to(device)

    optimizer = optim.AdamW(model.parameters(), lr=Learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Epochs)
    criterion_ft = FocalTverskyLoss()
    ema = EMA(model, decay=0.999)

    scaler = torch.cuda.amp.GradScaler(enabled=use_cuda and (not disable_amp))

    save_dir = os.path.join(project_root, "saved_models")
    os.makedirs(save_dir, exist_ok=True)
    best_dice = 0.0

    print(
        f"Training {model_type} | Batch: {Batch_size} | LR: {Learning_rate} | EMA Enabled"
    )

    for epoch in range(Epochs):
        model.train()
        loss_temp = 0.0
        optimizer.zero_grad(set_to_none=True)
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Epochs}")

        for batch_idx, (image, mask) in enumerate(progress_bar):
            image = image.to(device, non_blocking=use_cuda)
            mask = mask.to(device, non_blocking=use_cuda)

            with torch.cuda.amp.autocast(enabled=use_cuda and (not disable_amp)):
                out = model(image)
                loss = 0.5 * criterion_ft(out, mask) + 0.5 * dice_loss_from_logits(
                    out, mask
                )
                loss = loss / accumulation_steps

            if not torch.isfinite(loss):
                continue

            scaler.scale(loss).backward()

            if ((batch_idx + 1) % accumulation_steps == 0) or (
                (batch_idx + 1) == len(train_loader)
            ):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                ema.update()

            current_loss = loss.item() * accumulation_steps
            loss_temp += current_loss
            progress_bar.set_postfix({"Loss": f"{current_loss:.4f}"})

        scheduler.step()

        print(f"Evaluating Epoch {epoch+1}...")
        ema.apply_shadow()

        val_dice, best_th = evaluate(model, val_loader, device)
        print(
            f"Epoch [{epoch+1}/{Epochs}] - Train Loss: {loss_temp/len(train_loader):.4f} | "
            f"Val Dice: {val_dice:.4f} (Thresh: {best_th:.2f})"
        )

        if val_dice > best_dice:
            best_dice = val_dice
            save_path = os.path.join(save_dir, f"best_{model_type}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved (Dice: {best_dice:.4f})")

        ema.restore()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["UNet", "ResNet34_UNet"],
        default="ResNet34_UNet",
    )
    parser.add_argument("--disable_amp", action="store_true")
    args = parser.parse_args()

    train(
        args.epochs,
        args.batch_size,
        args.learning_rate,
        args.model_type,
        args.disable_amp,
    )
