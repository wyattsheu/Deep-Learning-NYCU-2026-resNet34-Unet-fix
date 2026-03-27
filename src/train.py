import os
import argparse
import platform
from contextlib import nullcontext
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchinfo import summary

from oxford_pet import OxfordPetDataset
from models.unet import UNet
from models.resnet34_unet import ResNet34_UNet
from evaluate import evaluate

# 🌟 從 utils 引入集成的 Losses
from utils import (
    focal_loss_from_logits,
    dice_loss_from_logits,
    boundary_loss_from_logits,
)

if platform.system() != "Windows":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "0")


def train(
    Epochs=60,
    Batch_size=24,
    Learning_rate=1e-4,
    model_type="ResNet34_UNet",
    show_summary=False,
    max_lr_mult=3.0,
    disable_amp=False,
):
    project_root = os.path.abspath(os.getcwd())
    data_dir = os.path.join(project_root, "dataset")

    if model_type == "UNet":
        IMG_SIZE = 572
        MASK_SIZE = 388
        accumulation_steps = 4  # 🌟 UNet 硬體受限 Batch=4，我們累積4步(實質 Batch=16)
    else:
        IMG_SIZE = 256
        MASK_SIZE = 256
        accumulation_steps = 1  # ResNet34 不需累積

    train_dataset = OxfordPetDataset(
        data_dir=data_dir, split="train", image_size=IMG_SIZE, mask_size=MASK_SIZE
    )
    val_dataset = OxfordPetDataset(
        data_dir=data_dir, split="val", image_size=IMG_SIZE, mask_size=MASK_SIZE
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    use_cuda = device.type == "cuda"

    if use_cuda:
        torch.backends.cudnn.benchmark = True

    train_loader = DataLoader(
        train_dataset,
        batch_size=Batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=use_cuda,
    )
    val_loader = DataLoader(val_dataset, batch_size=Batch_size, shuffle=False)

    model = UNet().to(device) if model_type == "UNet" else ResNet34_UNet().to(device)

    if show_summary:
        summary(model, input_size=(1, 3, IMG_SIZE, IMG_SIZE), device=device)

    print(
        f"device: {device}\ntraining by {model_type} model\nbatch size: {Batch_size} (Accumulation steps: {accumulation_steps})"
    )

    # 🌟 使用 AdamW 並給予明確的 Weight Decay 進行正則化防過擬合
    optimizer = optim.AdamW(model.parameters(), lr=Learning_rate, weight_decay=1e-2)
    max_lr = Learning_rate * max_lr_mult

    # 🌟 校正 Scheduler 的步數，並延長暖身期 (pct_start=0.2)
    total_steps_per_epoch = max(1, len(train_loader) // accumulation_steps)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        steps_per_epoch=total_steps_per_epoch,
        epochs=Epochs,
        pct_start=0.2,
    )

    amp_enabled = use_cuda and (not disable_amp)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    save_dir = os.path.join(project_root, "saved_models")
    os.makedirs(save_dir, exist_ok=True)
    best_dice = 0.0

    for epoch in range(Epochs):
        model.train()
        loss_temp = 0.0
        valid_steps = 0
        optimizer.zero_grad(set_to_none=True)
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Epochs}")

        for batch_idx, (image, mask) in enumerate(progress_bar):
            image = image.to(device, non_blocking=use_cuda)
            mask = mask.to(device, non_blocking=use_cuda)

            with torch.cuda.amp.autocast(enabled=amp_enabled):
                out = model(image)

                # 🌟 三重強效 Loss 融合 (包含邊界約束)
                focal_l = focal_loss_from_logits(out, mask)
                dice_l = dice_loss_from_logits(out, mask)
                bound_l = boundary_loss_from_logits(out, mask)

                if model_type == "UNet":
                    loss = 0.4 * focal_l + 0.4 * dice_l + 0.2 * bound_l
                else:
                    loss = 0.2 * focal_l + 0.6 * dice_l + 0.2 * bound_l

                # 🌟 梯度累積：將 Loss 除以累積步數
                loss = loss / accumulation_steps

            if not torch.isfinite(loss):
                continue

            scaler.scale(loss).backward()

            # 🌟 滿指定步數才進行權重更新
            if ((batch_idx + 1) % accumulation_steps == 0) or (
                (batch_idx + 1) == len(train_loader)
            ):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            # 顯示時將 Loss 乘回原來大小
            current_loss = loss.item() * accumulation_steps
            loss_temp += current_loss
            valid_steps += 1
            progress_bar.set_postfix({"Loss": f"{current_loss:.4f}"})

        avg_train_loss = loss_temp / max(1, valid_steps)
        print(f"Evaluating Epoch {epoch+1}...")
        val_dice = evaluate(model, val_loader, device)

        print(
            f"Epoch [{epoch+1}/{Epochs}] - Train Loss: {avg_train_loss:.4f} | Val Dice Score: {val_dice:.4f}"
        )

        if val_dice > best_dice:
            best_dice = val_dice
            save_path = os.path.join(save_dir, f"best_{model_type}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"new model saved at {save_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["UNet", "ResNet34_UNet"],
        default="ResNet34_UNet",
    )
    parser.add_argument("--show_summary", action="store_true")
    parser.add_argument("--max_lr_mult", type=float, default=3.0)
    parser.add_argument("--disable_amp", action="store_true")
    args = parser.parse_args()

    train(
        args.epochs,
        args.batch_size,
        args.learning_rate,
        args.model_type,
        args.show_summary,
        args.max_lr_mult,
        args.disable_amp,
    )
