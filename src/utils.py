import torch
import torch.nn.functional as F
import numpy as np
import scipy.ndimage as ndimage


def calculate_dice_score(pred, target):
    """計算 Dice Score 供評估使用 (平滑值改為安全的 1e-5)"""
    pred = pred.float()
    target = target.float()

    if pred.min() < 0 or pred.max() > 1:
        pred = torch.sigmoid(pred)

    pred = (pred > 0.5).float()
    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)

    intersection = (pred * target).sum(dim=1)
    denominator = pred.sum(dim=1) + target.sum(dim=1)
    dice = (2.0 * intersection + 1e-5) / (denominator + 1e-5)
    return dice.mean().item()


# ==========================================
# 🌟 損失函數區 (Loss Functions)
# ==========================================
def dice_loss_from_logits(logits, targets, smooth=1e-5):
    probs = torch.sigmoid(logits).float()
    targets = targets.float()
    probs = probs.view(probs.size(0), -1)
    targets = targets.view(targets.size(0), -1)
    intersection = (probs * targets).sum(dim=1)
    dice = (2.0 * intersection + smooth) / (
        probs.sum(dim=1) + targets.sum(dim=1) + smooth
    )
    return 1.0 - dice.mean()


def focal_loss_from_logits(logits, targets, alpha=0.25, gamma=2.0):
    bce_loss = F.binary_cross_entropy_with_logits(
        logits, targets.float(), reduction="none"
    )
    pt = torch.exp(-torch.clamp(bce_loss, max=100.0))
    focal_loss = alpha * (1 - pt) ** gamma * bce_loss
    return focal_loss.mean()


def boundary_loss_from_logits(logits, targets):
    """
    邊界損失 (Boundary Loss)：
    利用 MaxPool 產生形態學的膨脹與侵蝕，快速萃取預測機率與標籤的「邊緣輪廓」，
    強迫模型不能只對準面積，還要完美貼合形狀與邊緣。
    """
    probs = torch.sigmoid(logits)
    targets_float = targets.float()

    # 預測邊緣
    probs_dilated = F.max_pool2d(probs, kernel_size=3, stride=1, padding=1)
    probs_eroded = -F.max_pool2d(-probs, kernel_size=3, stride=1, padding=1)
    probs_boundary = probs_dilated - probs_eroded

    # 真實邊緣
    targets_dilated = F.max_pool2d(targets_float, kernel_size=3, stride=1, padding=1)
    targets_eroded = -F.max_pool2d(-targets_float, kernel_size=3, stride=1, padding=1)
    targets_boundary = targets_dilated - targets_eroded

    return F.mse_loss(probs_boundary, targets_boundary)


# ==========================================
# 🌟 後處理區 (Post-processing)
# ==========================================
def postprocess_batch_tensors(preds_tensor):
    """
    形態學後處理：填補孔洞 (Hole Filling) 與 保留最大連通域 (Largest Connected Component)
    能瞬間消除遠處的 False Positives (碎斑)，並讓腫瘤/器官實體更符合解剖學邏輯。
    """
    preds_np = (torch.sigmoid(preds_tensor) > 0.5).cpu().numpy()
    processed_np = np.zeros_like(preds_np)

    for i in range(preds_np.shape[0]):
        mask = preds_np[i, 0]
        # 1. 填補內部不合理的破洞
        mask_filled = ndimage.binary_fill_holes(mask)

        # 2. 保留最大的連通區塊 (假設每張圖只有一個主要實體)
        labeled, num_feat = ndimage.label(mask_filled)
        if num_feat > 1:
            sizes = ndimage.sum(mask_filled, labeled, range(1, num_feat + 1))
            largest = np.argmax(sizes) + 1
            processed_np[i, 0] = labeled == largest
        else:
            processed_np[i, 0] = mask_filled

    return torch.from_numpy(processed_np).to(preds_tensor.device).float()


# 你可以在這裡加入其他的輔助函式，例如視覺化預測結果的 function


def visualize_predictions(image, pred_mask, target_mask=None, save_path=None):
    """
    視覺化預測結果

    Args:
        image: 輸入圖像 (C, H, W) 或 (H, W)
        pred_mask: 預測的 mask (H, W)
        target_mask: 目標的 mask (H, W)，可選
        save_path: 保存圖像的路徑，如果為 None 則只顯示
    """
    import matplotlib.pyplot as plt

    # 轉換為 numpy
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.cpu().numpy()
    if isinstance(target_mask, torch.Tensor):
        target_mask = target_mask.cpu().numpy()

    # 處理圖像形狀
    if image.ndim == 3:
        image = np.transpose(image, (1, 2, 0))
        image = np.squeeze(image)

    # 建立圖表
    num_plots = 3 if target_mask is not None else 2
    fig, axes = plt.subplots(1, num_plots, figsize=(15, 5))

    if num_plots == 2:
        axes = [axes[0], axes[1]]

    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(pred_mask, cmap="gray")
    axes[1].set_title("Predicted Mask")
    axes[1].axis("off")

    if target_mask is not None:
        axes[2].imshow(target_mask, cmap="gray")
        axes[2].set_title("Ground Truth Mask")
        axes[2].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show(block=True)

    plt.close()


def visualize_predictions_grid(samples):
    """
    將多個預測結果畫在同一張圖上。

    Args:
        samples: list of tuples (image_id, image, pred_mask, target_mask)
    """
    if not samples:
        return

    import matplotlib.pyplot as plt

    num_samples = len(samples)
    has_target = samples[0][3] is not None
    num_cols = 3 if has_target else 2

    # Make a more compact canvas so the whole grid is easier to view on screen.
    cell_w = 3.2
    cell_h = 2.4
    fig, axes = plt.subplots(
        num_samples,
        num_cols,
        figsize=(cell_w * num_cols, cell_h * num_samples),
    )

    # 確保 axes 是二維的，即使只有一個樣本
    if num_samples == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, (image_id, image, pred_mask, target_mask) in enumerate(samples):
        # 轉換為 numpy
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        if isinstance(pred_mask, torch.Tensor):
            pred_mask = pred_mask.cpu().numpy()
        if isinstance(target_mask, torch.Tensor):
            target_mask = target_mask.cpu().numpy()

        # 處理圖像形狀
        if image.ndim == 3:
            image = np.transpose(image, (1, 2, 0))
            image = np.squeeze(image)

        axes[i, 0].imshow(image, cmap="gray")
        axes[i, 0].set_title(f"Image: {image_id}", fontsize=9)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(pred_mask, cmap="gray")
        axes[i, 1].set_title("Predicted Mask", fontsize=9)
        axes[i, 1].axis("off")

        if has_target and target_mask is not None:
            axes[i, 2].imshow(target_mask, cmap="gray")
            axes[i, 2].set_title("Ground Truth Mask", fontsize=9)
            axes[i, 2].axis("off")

    plt.tight_layout(pad=0.6, w_pad=0.4, h_pad=0.6)
    plt.show(block=True)
    plt.close()
