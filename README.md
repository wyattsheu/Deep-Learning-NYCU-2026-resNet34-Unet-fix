# Deep-Learning-NYCU-2026-resNet34-Unet

2026/3/23
CUDA 驗證與環境設置：

確認了 CUDA 是否正確安裝，並解決了 PyTorch 的相關警告（如 expandable_segments）。
UNet 訓練穩定性改進：

解決了訓練過程中的記憶體限制問題。
修復了梯度爆炸問題，加入了以下改進：
調整學習率調度策略（使用更安全的 OneCycleLR）。
啟用 AMP（混合精度訓練）以減少記憶體使用。
增加 NaN/Inf 檢測與處理，避免訓練中斷。

# 🚀 深度學習影像分割優化策略：從資料到推論的極致榨取

本文件詳細記錄了在不修改基礎模型架構（U-Net / ResNet34-UNet）的前提下，如何透過**資料增強 (Data Augmentation)**、**進階訓練策略 (Training Strategies)** 以及**推論優化 (Test-Time Augmentation, TTA)**，有效提升模型的泛化能力與最終的 Dice Score。

---

## 1. 🖼️ 資料增強 (Data Augmentation)

在醫學影像或少樣本的影像分割任務中，模型極易產生過擬合（Overfitting）。透過動態的資料增強，我們能讓模型在每個 Epoch 看到的影像都有微小的變化，強迫模型學習「物體的本質特徵」而非死背像素。

### 核心策略

- **幾何變換 (Geometric Transforms)**：使用水平翻轉 (Horizontal Flip) 與 隨機旋轉 (Random Rotation)
  - ⚠️ **關鍵細節**
    - Image 與 Mask 必須進行**完全一致的幾何變換**
    - Image 使用 `BILINEAR`
    - Mask 使用 `NEAREST`（避免標籤污染）

- **光學變換 (Color Jitter)**：隨機改變亮度與對比度
  - ⚠️ **關鍵細節**
    - 只能作用於 Image
    - **絕對不能套用在 Mask**

### 💻 實作程式碼 (`src/oxford_pet.py`)

```python
import random
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

if self.split == "train":
    # 1. 水平翻轉
    if random.random() > 0.5:
        image = TF.hflip(image)
        mask = TF.hflip(mask)

    # 2. 隨機旋轉
    if random.random() > 0.5:
        angle = random.uniform(-15.0, 15.0)
        image = TF.rotate(
            image, angle,
            interpolation=InterpolationMode.BILINEAR, fill=0
        )
        mask = TF.rotate(
            mask, angle,
            interpolation=InterpolationMode.NEAREST, fill=0
        )

    # 3. 顏色抖動（僅 image）
    if random.random() > 0.5:
        brightness_factor = random.uniform(0.8, 1.2)
        image = TF.adjust_brightness(image, brightness_factor)
        contrast_factor = random.uniform(0.8, 1.2)
        image = TF.adjust_contrast(image, contrast_factor)
```

---

## 2. 🏋️‍♂️ 訓練策略 (Training Strategies)

良好的訓練策略能幫助模型更穩定地收斂，並逃離局部最佳解（Local Minima）。

### 核心策略

- **AdamW 優化器**
  - 解耦 Weight Decay
  - 提升正則化效果

- **OneCycleLR 學習率排程**
  - 前期快速升高學習率（探索）
  - 後期逐步下降（精細收斂）

- **梯度裁剪 (Gradient Clipping)**
  - 防止梯度爆炸
  - 對無 BatchNorm 的 U-Net 特別重要

### 💻 實作程式碼 (`src/train.py`)

```python
import torch.optim as optim
import torch.nn.utils as utils

# 初始化
optimizer = optim.AdamW(
    model.parameters(),
    lr=Learning_rate,
    weight_decay=1e-2
)

scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=Learning_rate * 10,
    steps_per_epoch=len(train_loader),
    epochs=Epochs,
)

# 訓練過程
scaler.scale(loss).backward()

# AMP 下需先 unscale
scaler.unscale_(optimizer)
utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

scaler.step(optimizer)
scaler.update()
scheduler.step()
```

---

## 3. 🔮 推論優化 (Test-Time Augmentation, TTA)

TTA 是提升模型表現的「免費技巧」，不需要重新訓練模型即可提高準確率。

### 核心策略

- 原圖預測 + 翻轉圖預測
- 翻轉結果再對齊回來
- 最後取平均

👉 優點：

- 降低單一視角偏差
- 提升邊界穩定性
- 提升 Dice Score

### 💻 實作程式碼 (`src/inference.py`)

```python
import torch

with torch.no_grad():
    # 原始預測
    logits_orig = model(images)
    probs_orig = torch.sigmoid(logits_orig)

    # TTA：水平翻轉
    images_flipped = torch.flip(images, dims=[3])
    logits_flipped = model(images_flipped)
    probs_flipped = torch.sigmoid(logits_flipped)

    # 翻轉回來
    probs_flipped_back = torch.flip(probs_flipped, dims=[3])

    # 平均
    final_probs = (probs_orig + probs_flipped_back) / 2.0

    # 二值化
    preds = (final_probs > 0.5).float().cpu()
```

---

## ✅ 總結

在**不改動模型架構**的前提下，本策略透過三個層面達成最佳化：

| 層面 | 技術                                   |
| ---- | -------------------------------------- |
| 資料 | Data Augmentation                      |
| 訓練 | AdamW + OneCycleLR + Gradient Clipping |
| 推論 | Test-Time Augmentation                 |

👉 三者協同效果：

- 提升泛化能力
- 減少過擬合
- 穩定訓練過程
- 最大化模型表現（Dice Score）

---

📌 適用場景：

- 醫學影像分割
- 小資料集任務
- Kaggle / 競賽優化
- 工業影像檢測
