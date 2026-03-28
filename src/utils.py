import torch
import torch.nn as nn
import numpy as np
import scipy.ndimage as ndimage


# ==========================================
# 1. Top-tier Loss (Focal Tversky Loss)
# ==========================================
class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75, smooth=1e-6):
        """
        alpha: penalty for false positives
        beta: penalty for false negatives
        gamma: focal modulation factor
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits, targets):
        preds = torch.sigmoid(logits)
        preds = preds.view(-1)
        targets = targets.float().view(-1)

        tp = (preds * targets).sum()
        fp = ((1 - targets) * preds).sum()
        fn = (targets * (1 - preds)).sum()

        tversky = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )
        return (1 - tversky) ** self.gamma


def dice_loss_from_logits(logits, targets, smooth=1e-5):
    probs = torch.sigmoid(logits).float()
    targets = targets.float().view(targets.size(0), -1)
    probs = probs.view(probs.size(0), -1)
    intersection = (probs * targets).sum(dim=1)
    dice = (2.0 * intersection + smooth) / (
        probs.sum(dim=1) + targets.sum(dim=1) + smooth
    )
    return 1.0 - dice.mean()


# ==========================================
# 2. Exponential Moving Average (EMA)
# ==========================================
class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_avg = (
                    self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
                )
                self.shadow[name] = new_avg.clone()

    def apply_shadow(self):
        """Apply EMA weights before validation/checkpoint."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        """Restore training weights after validation/checkpoint."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


# ==========================================
# 3. Evaluation and post-processing
# ==========================================
def calculate_dice_score(probs, targets, threshold=0.5):
    """Calculate Dice score from sigmoid probabilities."""
    preds = (probs > threshold).float()
    preds = preds.view(preds.size(0), -1)
    targets = targets.float().view(targets.size(0), -1)

    intersection = (preds * targets).sum(dim=1)
    denominator = preds.sum(dim=1) + targets.sum(dim=1)
    dice = (2.0 * intersection + 1e-5) / (denominator + 1e-5)
    return dice.mean().item()


def postprocess_batch_tensors(preds_tensor, threshold=0.5):
    """Morphological post-processing with hole fill + largest component."""
    if preds_tensor.max() > 1.0 or preds_tensor.min() < 0.0:
        preds_tensor = torch.sigmoid(preds_tensor)

    preds_np = (preds_tensor > threshold).cpu().numpy()
    processed_np = np.zeros_like(preds_np)

    for i in range(preds_np.shape[0]):
        mask = preds_np[i, 0]
        mask_filled = ndimage.binary_fill_holes(mask)
        labeled, num_feat = ndimage.label(mask_filled)
        if num_feat > 1:
            sizes = ndimage.sum(mask_filled, labeled, range(1, num_feat + 1))
            largest = np.argmax(sizes) + 1
            processed_np[i, 0] = labeled == largest
        else:
            processed_np[i, 0] = mask_filled

    return torch.from_numpy(processed_np).to(preds_tensor.device).float()


def visualize_predictions(image, pred_mask, target_mask=None, save_path=None):
    """
    Visualize prediction results.

    Args:
        image: input image tensor/array
        pred_mask: predicted mask
        target_mask: ground-truth mask, optional
        save_path: output path, optional
    """
    import matplotlib.pyplot as plt

    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.cpu().numpy()
    if isinstance(target_mask, torch.Tensor):
        target_mask = target_mask.cpu().numpy()

    if image.ndim == 3:
        image = np.transpose(image, (1, 2, 0))
        image = np.squeeze(image)

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
    Plot multiple prediction results on one figure.

    Args:
        samples: list of tuples (image_id, image, pred_mask, target_mask)
    """
    if not samples:
        return

    import matplotlib.pyplot as plt

    num_samples = len(samples)
    has_target = samples[0][3] is not None
    num_cols = 3 if has_target else 2

    cell_w = 3.2
    cell_h = 2.4
    fig, axes = plt.subplots(
        num_samples,
        num_cols,
        figsize=(cell_w * num_cols, cell_h * num_samples),
    )

    if num_samples == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, (image_id, image, pred_mask, target_mask) in enumerate(samples):
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        if isinstance(pred_mask, torch.Tensor):
            pred_mask = pred_mask.cpu().numpy()
        if isinstance(target_mask, torch.Tensor):
            target_mask = target_mask.cpu().numpy()

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
