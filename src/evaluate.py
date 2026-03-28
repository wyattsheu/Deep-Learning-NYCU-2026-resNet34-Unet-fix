import torch
from utils import calculate_dice_score, postprocess_batch_tensors


def evaluate(model, dataloader, device):
    model.eval()

    thresholds = [i / 100.0 for i in range(30, 75, 5)]
    total_dice = {t: 0.0 for t in thresholds}
    total_samples = 0

    with torch.no_grad():
        for image, mask in dataloader:
            image = image.to(device)
            mask = mask.to(device)
            logits = model(image)
            probs = torch.sigmoid(logits)

            batch_size = image.size(0)
            total_samples += batch_size

            for threshold in thresholds:
                processed_preds = postprocess_batch_tensors(probs, threshold=threshold)
                batch_dice = calculate_dice_score(processed_preds, mask, threshold=0.5)
                total_dice[threshold] += batch_dice * batch_size

    if total_samples == 0:
        return 0.0, 0.5

    best_threshold = 0.5
    best_dice = 0.0
    for threshold in thresholds:
        mean_dice = total_dice[threshold] / total_samples
        if mean_dice > best_dice:
            best_dice = mean_dice
            best_threshold = threshold

    return best_dice, best_threshold
