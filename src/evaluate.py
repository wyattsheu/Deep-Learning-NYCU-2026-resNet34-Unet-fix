import torch
from utils import calculate_dice_score


def evaluate(model, dataloader, device):
    """
    評估模型在驗證集上的表現。
    這裡我們不加入形態學後處理，以反映模型神經網路本身的真實輸出能力，
    方便判斷模型是否真的學到了好特徵。後處理統一交由 inference.py 在最後生成 Kaggle 提交檔時處理。
    """
    model.eval()
    total_dice = 0.0
    total_samples = 0

    with torch.no_grad():
        for image, mask in dataloader:
            image = image.to(device)
            mask = mask.to(device)

            # 預測輸出 logits
            pred_logits = model(image)

            # 直接計算 dice score，不經過 postprocess_batch_tensors
            batch_dice = calculate_dice_score(pred_logits, mask)

            batch_size = image.size(0)
            total_dice += batch_dice * batch_size
            total_samples += batch_size

    if total_samples == 0:
        return 0.0

    return total_dice / total_samples
