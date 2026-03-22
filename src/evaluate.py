import torch
from utils import calculate_dice_score


def evaluate(model, dataloader, device):
    """
    TODO:
    1. 將模型設為 eval 模式 (model.eval())。
    2. 關閉梯度計算 (torch.no_grad())。
    3. 跑過整個 validation dataloader，計算平均的 Loss 與 Dice Score。
    """
    model.eval()
    total_dice = 0.0
    total_samples = 0

    with torch.no_grad():
        for image, mask in dataloader:
            image = image.to(device)
            mask = mask.to(device)

            pred = model(image)
            batch_dice = calculate_dice_score(pred, mask)

            batch_size = image.size(0)
            total_dice += batch_dice * batch_size
            total_samples += batch_size

    if total_samples == 0:
        return 0.0

    return total_dice / total_samples
