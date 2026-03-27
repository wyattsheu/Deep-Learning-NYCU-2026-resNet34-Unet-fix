import os
import random


import torch
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from torchvision.transforms import v2
from datasets import load_dataset


def apply_clahe(pil_img):
    """
    對 PIL 影像應用 CLAHE (限制對比度自適應直方圖均衡化)。
    只強化亮度 (L) 通道，凸顯陰影或模糊邊界，且不會讓毛色失真。
    """
    img_np = np.array(pil_img)
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    # 建立 CLAHE 物件 (clipLimit=2.0 是溫和且有效的經驗值)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    limg = cv2.merge((cl, a, b))
    img_clahe = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    return Image.fromarray(img_clahe)


def add_gaussian_noise(pil_img, sigma=0.05):
    """只對影像加入輕微高斯雜訊，避免破壞 segmentation mask 標籤。"""
    img_tensor = TF.to_tensor(pil_img)
    noise = torch.randn_like(img_tensor) * sigma
    img_tensor = torch.clamp(img_tensor + noise, 0.0, 1.0)
    return TF.to_pil_image(img_tensor)


class LetterBoxResize:
    def __init__(self, target_size, interpolation=InterpolationMode.BILINEAR, fill=0):
        self.target_size = target_size
        self.interpolation = interpolation
        self.fill = fill

    def __call__(self, img):
        w, h = img.size
        scale = self.target_size / max(h, w)

        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))

        resized = TF.resize(
            img,
            (new_h, new_w),
            interpolation=self.interpolation,
        )

        pad_left = (self.target_size - new_w) // 2
        pad_top = (self.target_size - new_h) // 2
        pad_right = self.target_size - new_w - pad_left
        pad_bottom = self.target_size - new_h - pad_top

        return TF.pad(
            resized,
            (pad_left, pad_top, pad_right, pad_bottom),
            fill=self.fill,
        )


class OxfordPetDataset(Dataset):
    HF_DATASET_NAME = "wyattsheu/oxford-pet-full-raw"

    # 🌟 新增 image_size 與 mask_size，預設為 UNet 的尺寸
    def __init__(
        self,
        data_dir="dataset",
        split="train",
        image_size=572,
        mask_size=388,
        return_mask_for_test=False,
        return_unpadded_for_test=False,
    ):
        self.split = split
        self.data_dir = os.path.abspath(data_dir)
        self.image_size = image_size
        self.mask_size = mask_size
        self.return_mask_for_test = return_mask_for_test
        self.return_unpadded_for_test = return_unpadded_for_test

        txt_path = os.path.join(self.data_dir, f"{split}.txt")
        if not os.path.exists(txt_path):
            raise FileNotFoundError(f"Split file not found: {txt_path}")

        with open(txt_path, "r", encoding="utf-8") as f:
            self.target_names = []
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                self.target_names.append(line.split()[0])

        self.use_local = False
        self.local_root = None
        candidate_roots = [
            os.path.join(self.data_dir, "oxford-iiit-pet"),
            self.data_dir,
        ]
        for root in candidate_roots:
            images_dir = os.path.join(root, "images")
            trimaps_dir = os.path.join(root, "annotations", "trimaps")
            if os.path.isdir(images_dir) and os.path.isdir(trimaps_dir):
                self.use_local = True
                self.local_root = root
                break

        if self.use_local:
            print(f"使用本地資料夾: {self.local_root}")
        else:
            print(f"正在從雲端索引完整資料庫 ({split})...")
            raw_ds = load_dataset(self.HF_DATASET_NAME)

            if isinstance(raw_ds, dict):
                if "train" in raw_ds:
                    full_ds = raw_ds["train"]
                else:
                    first_split = next(iter(raw_ds.keys()))
                    full_ds = raw_ds[first_split]
            else:
                full_ds = raw_ds

            self.name_to_idx = {name: i for i, name in enumerate(full_ds["filename"])}
            self.ds = full_ds

        # 🌟 計算需要鏡像填充的像素大小 (例如 572 - 388 = 184，除以2 = 92)
        pad_size = (self.image_size - self.mask_size) // 2

        # 先做等比例縮放 + 黑邊補齊
        self.letterbox_image = LetterBoxResize(
            self.mask_size,
            interpolation=InterpolationMode.BILINEAR,
            fill=0,
        )
        self.letterbox_mask = LetterBoxResize(
            self.mask_size,
            interpolation=InterpolationMode.NEAREST,
            fill=0,
        )

        # 圖片後處理：補齊外圍犧牲區 (UNet 需要) 並轉 Tensor
        self.pad_and_tensor = transforms.Compose(
            [
                transforms.Pad(pad_size, padding_mode="reflect"),
                transforms.ToTensor(),
            ]
        )
        self.just_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.target_names)

    def __getitem__(self, idx):
        file_name = self.target_names[idx]

        if self.use_local:
            image_path = os.path.join(self.local_root, "images", file_name + ".jpg")
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Local image not found: {image_path}")
            image = Image.open(image_path).convert("RGB")
        else:
            hf_idx = self.name_to_idx[file_name]
            item = self.ds[hf_idx]
            image = item["image"].convert("RGB")

        # 記錄原始尺寸，供 test/inference 還原使用
        orig_w, orig_h = image.size

        # 效能關鍵：先降解析度，再做後續 augmentation
        image = self.letterbox_image(image)

        if self.split.startswith("test") and not self.return_mask_for_test:
            image_tensor = self.pad_and_tensor(image)
            image_unpadded_tensor = self.just_tensor(image)
            if self.return_unpadded_for_test:
                return (
                    image_tensor,
                    file_name,
                    torch.tensor([orig_h, orig_w]),
                    image_unpadded_tensor,
                )
            return image_tensor, file_name, torch.tensor([orig_h, orig_w])

        # 處理 Mask (輸出為 388x388)
        if self.use_local:
            mask_path = os.path.join(
                self.local_root, "annotations", "trimaps", file_name + ".png"
            )
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Local mask not found: {mask_path}")
            mask = Image.open(mask_path).convert("L")
        else:
            mask = item["mask"].convert("L")

        mask = self.letterbox_mask(mask)

        if self.split == "train":
            # 1. 安全的翻轉 (維持 50%)
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            # -----------------------------------------
            # 🌟 第二組：輕度幾何變形 (60%機率執行其一，40%維持原狀)
            # -----------------------------------------
            geom_choice = random.random()
            if geom_choice < 0.2:
                # 只做 ±10 度的微小旋轉
                angle = random.uniform(-10.0, 10.0)
                image = TF.rotate(
                    image, angle, interpolation=InterpolationMode.BILINEAR, fill=0
                )
                mask = TF.rotate(
                    mask, angle, interpolation=InterpolationMode.NEAREST, fill=0
                )

            elif geom_choice < 0.4:
                # 輕微平移與縮放 (0.95 ~ 1.05)
                affine = v2.RandomAffine(
                    degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)
                )
                image, mask = affine(image, mask)

            elif geom_choice < 0.6:
                # 微量彈性變形 (模擬組織蠕動，alpha值縮小)
                w, h = image.size
                base_size = max(w, h)
                elastic = v2.ElasticTransform(
                    alpha=base_size * 0.5, sigma=base_size * 0.02
                )
                image, mask = elastic(image, mask)

            # -----------------------------------------
            # 🌟 第三組：重度色彩光學 (80%機率執行其一，20%維持原狀)
            # 不影響解剖形狀，逼迫模型學習深層特徵
            # -----------------------------------------
            color_choice = random.random()
            if color_choice < 0.2:
                image = apply_clahe(image)
            elif color_choice < 0.4:
                image = TF.adjust_brightness(image, random.uniform(0.8, 1.2))
                image = TF.adjust_contrast(image, random.uniform(0.8, 1.2))
            elif color_choice < 0.6:
                image = add_gaussian_noise(image, sigma=0.05)
            elif color_choice < 0.8:
                blur = v2.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))
                image = blur(image)

        # 最後補邊與轉 tensor
        image_tensor = self.pad_and_tensor(image)
        image_unpadded_tensor = self.just_tensor(image)

        mask_array = np.array(mask)
        binary_mask = np.zeros_like(mask_array, dtype=np.float32)
        binary_mask[mask_array == 1] = 1.0

        mask_tensor = torch.from_numpy(binary_mask).unsqueeze(0)

        if self.split.startswith("test") and self.return_mask_for_test:
            if self.return_unpadded_for_test:
                return (
                    image_tensor,
                    file_name,
                    mask_tensor,
                    torch.tensor([orig_h, orig_w]),
                    image_unpadded_tensor,
                )
            return image_tensor, file_name, mask_tensor, torch.tensor([orig_h, orig_w])

        # 最終回傳：image_tensor(572x572), mask_tensor(388x388)
        return image_tensor, mask_tensor


def _find_local_oxford_root(data_dir):
    candidate_roots = [
        os.path.join(data_dir, "oxford-iiit-pet"),
        data_dir,
    ]
    for root in candidate_roots:
        images_dir = os.path.join(root, "images")
        trimaps_dir = os.path.join(root, "annotations", "trimaps")
        if os.path.isdir(images_dir) and os.path.isdir(trimaps_dir):
            return root
    return None


def _load_one_sample_for_visualization(data_dir, split="train"):
    txt_path = os.path.join(data_dir, f"{split}.txt")
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"Split file not found: {txt_path}")

    file_name = None
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            file_name = line.split()[0]
            break

    if file_name is None:
        raise RuntimeError(f"No valid sample found in split file: {txt_path}")

    local_root = _find_local_oxford_root(data_dir)
    if local_root is None:
        raise FileNotFoundError(
            "找不到本地 Oxford Pet 資料夾，請確認 dataset/oxford-iiit-pet 存在。"
        )

    image_path = os.path.join(local_root, "images", file_name + ".jpg")
    mask_path = os.path.join(local_root, "annotations", "trimaps", file_name + ".png")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Local image not found: {image_path}")
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Local mask not found: {mask_path}")

    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    return file_name, image, mask


def _tensor_to_hwc_uint8(tensor):
    tensor_arr = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    tensor_arr = np.clip(tensor_arr, 0.0, 1.0)
    return (tensor_arr * 255).astype(np.uint8)


def _visualize_all_augmentations():
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = [
        "Microsoft JhengHei",
        "PingFang TC",
        "SimHei",
        "Arial Unicode MS",
    ]
    plt.rcParams["axes.unicode_minus"] = False  # 避免設定字體後，負號變成方框的問題

    random.seed(42)
    torch.manual_seed(42)

    vis_data_dir = os.path.abspath("dataset")
    vis_file_name, vis_image, _ = _load_one_sample_for_visualization(
        vis_data_dir, split="train"
    )

    vis_angle = 12.0
    vis_brightness_factor = 1.15
    vis_contrast_factor = 1.15
    vis_noise_sigma = 0.05

    # 🌟 1. 提早建立 Dataset 實例，取得 LetterBox 工具
    vis_ds = OxfordPetDataset(
        data_dir=vis_data_dir,
        split="train",
        image_size=572,
        mask_size=388,
    )

    # 🌟 2. 關鍵修改：先將圖片縮放到模型實際訓練時的解析度 (例如 388x388)
    vis_image_small = vis_ds.letterbox_image(vis_image)

    # 🌟 3. 所有的 Augmentation 都改套用在 "vis_image_small" 上
    image_hflip = TF.hflip(vis_image_small)

    image_rotate = TF.rotate(
        vis_image_small,
        vis_angle,
        interpolation=InterpolationMode.BILINEAR,
        fill=0,
    )

    image_clahe = apply_clahe(vis_image_small)

    # Elastic 也是根據小圖的 size 動態計算
    w, h = vis_image_small.size
    print(f"Resized Image Size: {w}x{h}")

    base_size = max(w, h)
    # 這裡的 alpha/sigma 會自動配合 388 (或 256) 的尺寸算出剛好的力道
    elastic_alpha = base_size * 1.25
    elastic_sigma = base_size * 0.03
    print(
        f"Elastic Transform Parameters (Scaled): alpha={elastic_alpha:.1f}, sigma={elastic_sigma:.1f}"
    )

    elastic = v2.ElasticTransform(alpha=elastic_alpha, sigma=elastic_sigma)
    image_elastic, _ = elastic(vis_image_small, vis_image_small)

    affine = v2.RandomAffine(
        degrees=0,
        translate=(0.1, 0.1),
        scale=(0.9, 1.1),
    )
    image_affine = affine(vis_image_small)

    image_bc = TF.adjust_brightness(vis_image_small, vis_brightness_factor)
    image_bc = TF.adjust_contrast(image_bc, vis_contrast_factor)

    image_noise = add_gaussian_noise(vis_image_small, sigma=vis_noise_sigma)

    # 最終轉 Tensor 並加上鏡像外圍
    image_final = _tensor_to_hwc_uint8(vis_ds.pad_and_tensor(vis_image_small))
    image_unpadded = _tensor_to_hwc_uint8(vis_ds.just_tensor(vis_image_small))

    fig, axes = plt.subplots(2, 5, figsize=(24, 10))
    axes = axes.ravel()

    # 🌟 4. 調整顯示順序，讓你在第 2 格就能看到縮放後的基準圖
    items = [
        (np.array(vis_image), "Original Image 原圖(大尺寸)"),
        (image_unpadded, "LetterBox Resize 縮放後基準圖"),
        (np.array(image_hflip), "HFlip Image 水平翻轉"),
        (np.array(image_rotate), f"Rotate {vis_angle:.1f}° 旋轉"),
        (np.array(image_clahe), "CLAHE 增強對比"),
        (np.array(image_elastic), "Elastic Image 彈性變形"),
        (np.array(image_affine), "Translation + Zoom 平移+縮放"),
        (
            np.array(image_bc),
            f"Brightness {vis_brightness_factor:.2f} + Contrast {vis_contrast_factor:.2f}",
        ),
        (
            np.array(image_noise),
            f"Gaussian Noise 高斯雜訊 (sigma={vis_noise_sigma:.2f})",
        ),
        (image_final, "LetterBox + Reflect Pad (image_size) 最終填充"),
    ]

    for ax, (arr, title) in zip(axes, items):
        ax.imshow(arr)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    for ax in axes[len(items) :]:
        ax.axis("off")

    fig.suptitle(
        f"Oxford Pet Augmentation Visualization: {vis_file_name} 實際訓練視野",
        fontsize=16,
    )
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    plt.show()


if __name__ == "__main__":

    _visualize_all_augmentations()
