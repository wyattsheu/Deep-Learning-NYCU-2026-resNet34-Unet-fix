import os

# 加入這行來減少 CUDA 記憶體破碎化
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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


def dice_loss_from_logits(logits, targets, smooth=1.0):
    probs = torch.sigmoid(logits).float()
    targets = targets.float()

    probs = probs.view(probs.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    intersection = (probs * targets).sum(dim=1)
    dice = (2.0 * intersection + smooth) / (
        probs.sum(dim=1) + targets.sum(dim=1) + smooth
    )
    return 1.0 - dice.mean()


def train():
    Epochs = 50
    Batch_size = 32 if torch.cuda.device_count() > 1 else 16
    Learning_rate = 1e-4
    model_type = "UNet"  # 可選擇 "UNet" 或 "ResNet34_UNet"

    project_root = os.path.abspath(os.getcwd())
    data_dir = os.path.join(project_root, "dataset")

    # 🌟 動態設定尺寸，保證兩個模型都能相容
    if model_type == "UNet":
        IMG_SIZE = 572
        MASK_SIZE = 388
    else:
        # 假設 ResNet34_UNet 是有 padding 的卷積，輸入與輸出相同
        IMG_SIZE = 256
        MASK_SIZE = 256

    # 🌟 傳入指定的尺寸給 Dataset
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

    if model_type == "UNet":
        model = UNet().to(device)
    else:
        model = ResNet34_UNet().to(device)

    print("\n" + "=" * 60)
    print(f"🔍 正在掃描 {model_type} 模型內部架構...")
    print("=" * 60)
    # 這裡的 IMG_SIZE 會自動抓取你在上面設定的 572 (UNet) 或 256 (ResNet)
    summary(
        model,
        input_size=(1, 3, IMG_SIZE, IMG_SIZE),
        col_names=["input_size", "output_size", "num_params", "kernel_size"],
        depth=4,
    )
    print("=" * 60 + "\n")

    print(f"device: {device}")
    print(f"training by {model_type} model")
    ###########應註解掉
    # if hasattr(torch, "compile"):
    #     try:
    #         model = torch.compile(model, mode="reduce-overhead")
    #         print("torch.compile enabled")
    #     except Exception as e:
    #         print(f"torch.compile skipped: {e}")

    # # 如果有多個 GPU，使用 nn.DataParallel
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with nn.DataParallel")
        model = nn.DataParallel(model)
    ###########

    bce_loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=Learning_rate,
        weight_decay=1e-2,
    )
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Learning_rate * 10,
        steps_per_epoch=len(train_loader),
        epochs=Epochs,
    )

    amp_enabled = use_cuda
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        except TypeError:
            scaler = torch.amp.GradScaler(enabled=amp_enabled)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    save_dir = os.path.join(project_root, "saved_models")
    os.makedirs(save_dir, exist_ok=True)
    best_dice = 0.0

    for epoch in range(Epochs):
        model.train()
        loss_temp = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Epochs}")

        for image, mask in progress_bar:
            image = image.to(device, non_blocking=use_cuda)
            mask = mask.to(device, non_blocking=use_cuda)

            optimizer.zero_grad(set_to_none=True)

            if amp_enabled and hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
                autocast_context = torch.amp.autocast(device_type="cuda", enabled=True)
            elif amp_enabled:
                autocast_context = torch.cuda.amp.autocast(enabled=True)
            else:
                autocast_context = nullcontext()

            with autocast_context:

                # 如果模型有被包裝 (有 module 屬性)，就脫殼；如果沒有，就保持原樣
                raw_model = model.module if hasattr(model, "module") else model
                out = raw_model(image)
                ###
                # out = model(image)
                bce_loss = bce_loss_fn(out, mask)
                dice_loss = dice_loss_from_logits(out, mask)
                loss = 0.2 * bce_loss + 0.8 * dice_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            loss_temp += loss.item()
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

        avg_train_loss = loss_temp / len(train_loader)

        print(f"Evaluating Epoch {epoch+1}...")
        val_dice = evaluate(model, val_loader, device)

        print(
            f"Epoch [{epoch+1}/{Epochs}] - Train Loss: {avg_train_loss:.4f} | Val Dice Score: {val_dice:.4f}"
        )

        if val_dice > best_dice:
            best_dice = val_dice

            save_path = os.path.join(save_dir, f"best_{model_type}.pth")
            # 🔥 神奇的一行：自動適應單卡/多卡環境
            model_to_save = model.module if hasattr(model, "module") else model
            # torch.save(model.state_dict(), save_path)
            torch.save(model_to_save.state_dict(), save_path)
            print(f"new model saved at {save_path}\n")


if __name__ == "__main__":
    train()
