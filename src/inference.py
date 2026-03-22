import argparse
import csv
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from oxford_pet import OxfordPetDataset
from models.resnet34_unet import ResNet34_UNet
from models.unet import UNet
from utils import (
    calculate_dice_score,
    visualize_predictions_grid,
)


try:
    NEAREST_RESAMPLE = Image.Resampling.NEAREST
except AttributeError:
    NEAREST_RESAMPLE = Image.NEAREST if hasattr(Image, "NEAREST") else 0


def get_model_io_size(model_type: str):
    if model_type == "UNet":
        return (572, 572), (388, 388)
    # ResNet34_UNet is trained with same-size input/output in this project.
    return (512, 512), (512, 512)


def infer_model_type_from_checkpoint(model_path: str):
    name = os.path.basename(model_path).lower()
    if "resnet34_unet" in name or name == "resnet34_unet.pth":
        return "ResNet34_UNet"
    if "unet" in name:
        return "UNet"
    return None


def auto_pick_checkpoint(model_path: str, model_type: str):
    if model_path:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
        inferred = infer_model_type_from_checkpoint(model_path)
        final_type = model_type if model_type else inferred
        return model_path, final_type

    candidates = [
        ("saved_models/best_ResNet34_UNet.pth", "ResNet34_UNet"),
        ("saved_models/ResNet34_UNet.pth", "ResNet34_UNet"),
        ("saved_models/best_UNet.pth", "UNet"),
        ("saved_models/UNet.pth", "UNet"),
    ]

    for path, picked_type in candidates:
        if os.path.exists(path) and (not model_type or model_type == picked_type):
            return path, picked_type

    raise FileNotFoundError(
        "Cannot find any checkpoint in saved_models/. "
        "Checked: " + ", ".join(path for path, _ in candidates)
    )


def center_crop_mask(mask: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    h, w = mask.shape
    top = (h - target_h) // 2
    left = (w - target_w) // 2
    return mask[top : top + target_h, left : left + target_w]


def rle_encode(mask: np.ndarray) -> str:
    """Encode a binary mask to RLE using column-major (Fortran) order."""
    pixels = mask.astype(np.uint8).flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def load_image_ids(data_dir: str, split_file: str = None, model_type: str = "UNet"):
    """Load image IDs from split file (first token per line)."""
    candidates = []
    if split_file:
        candidates.append(split_file)

    if model_type == "UNet":
        model_default_splits = [
            os.path.join(os.path.dirname(data_dir), "test_unet.txt"),
            os.path.join(os.path.dirname(data_dir), "test_res_unet.txt"),
        ]
    else:
        model_default_splits = [
            os.path.join(os.path.dirname(data_dir), "test_res_unet.txt"),
            os.path.join(os.path.dirname(data_dir), "test_unet.txt"),
        ]

    candidates.extend(model_default_splits)

    chosen_path = None
    for path in candidates:
        if os.path.exists(path):
            chosen_path = path
            break

    if chosen_path is None:
        raise FileNotFoundError(
            "Cannot find test split file. Checked: " + ", ".join(candidates)
        )

    image_ids = []
    with open(chosen_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            image_ids.append(line.split()[0])

    if len(image_ids) == 0:
        raise ValueError(f"No image IDs found in split file: {chosen_path}")

    return image_ids, chosen_path


def verify_output_shape(model, model_type: str, input_size, target_size, device):
    with torch.no_grad():
        dummy = torch.zeros(1, 3, input_size[0], input_size[1], device=device)
        out = model(dummy)
    actual = tuple(out.shape[-2:])
    if actual != target_size:
        raise ValueError(
            f"{model_type} output shape mismatch. Expected {target_size}, got {actual}."
        )


def validate_submission_rows(rows, expected_ids):
    """Validate basic Kaggle-format constraints and return issue list."""
    issues = []

    row_ids = [row[0] for row in rows]
    expected_set = set(expected_ids)
    row_set = set(row_ids)

    missing = sorted(expected_set - row_set)
    extra = sorted(row_set - expected_set)

    if missing:
        issues.append(f"Missing image_ids: {len(missing)}")
    if extra:
        issues.append(f"Unknown image_ids: {len(extra)}")
    if len(row_ids) != len(row_set):
        issues.append("Duplicated image_id found in submission")

    for image_id, encoded_mask in rows:
        if not image_id:
            issues.append("Found empty image_id")
            break
        if encoded_mask and any(ch not in "0123456789 " for ch in encoded_mask):
            issues.append(f"Invalid RLE format detected for image_id={image_id}")
            break

    return issues


def run_inference(args):
    model_type = args.model_type
    model_path = args.model_path
    data_dir = args.data_dir
    hf_dataset_name = args.hf_dataset_name
    hf_split = args.hf_split
    batch_size = args.batch_size
    submission_path = args.submission_path
    vis_dir = args.vis_dir
    threshold = args.threshold
    num_vis = args.num_vis

    os.makedirs(os.path.dirname(submission_path) or ".", exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    model_path, detected_type = auto_pick_checkpoint(model_path, model_type)
    model_type = detected_type
    input_size, target_size = get_model_io_size(model_type)

    if model_type == "UNet":
        model = UNet().to(device)
    else:
        model = ResNet34_UNet().to(device)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    verify_output_shape(model, model_type, input_size, target_size, device)

    # Inference 統一走 OxfordPetDataset（內部由 HF load_dataset 取圖），不再讀本地 jpg。
    if hf_dataset_name:
        OxfordPetDataset.HF_DATASET_NAME = hf_dataset_name

    split_name = (
        hf_split
        if hf_split
        else ("test_unet" if model_type == "UNet" else "test_res_unet")
    )
    if not split_name.startswith("test"):
        raise ValueError("For inference, split must be a test split (e.g. test_unet).")

    split_dir = data_dir
    split_path = os.path.join(split_dir, f"{split_name}.txt")
    if not os.path.exists(split_path):
        parent_dir = os.path.dirname(split_dir)
        parent_split_path = os.path.join(parent_dir, f"{split_name}.txt")
        if os.path.exists(parent_split_path):
            split_dir = parent_dir
            split_path = parent_split_path
        else:
            raise FileNotFoundError(
                f"Cannot find split file for inference: {split_path} or {parent_split_path}"
            )

    print(f"inferencing with model: {model_path} (detected type: {model_type})")
    print(f"Using device: {device}")

    test_dataset = OxfordPetDataset(
        data_dir=split_dir,
        split=split_name,
        image_size=input_size[0],
        mask_size=target_size[0],
        return_mask_for_test=True,
        return_unpadded_for_test=True,
    )
    image_ids = test_dataset.target_names
    gt_available = True

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    submissions = []
    total_dice = 0.0
    total_count = 0
    vis_buffer = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            if gt_available:
                images, batch_image_ids, gt_masks, orig_sizes, vis_images = batch
                gt_masks = gt_masks.to(device)
            else:
                images, batch_image_ids, orig_sizes, vis_images = batch
                gt_masks = None

            images = images.to(device)
            # logits = model(images)
            # probs = torch.sigmoid(logits)
            # preds = (probs > threshold).float()

            # 原始預測
            logits = model(images)
            probs1 = torch.sigmoid(logits)

            # TTA: 水平翻轉預測
            images_flipped = torch.flip(images, dims=[3])
            logits_flipped = model(images_flipped)
            probs_flipped = torch.sigmoid(logits_flipped)
            probs2 = torch.flip(probs_flipped, dims=[3])

            # 綜合兩次預測結果（平均）
            final_probs = (probs1 + probs2) / 2.0
            preds = (final_probs > threshold).float()

            if gt_available:
                batch_dice = calculate_dice_score(preds, gt_masks)
                batch_size_actual = images.size(0)
                total_dice += batch_dice * batch_size_actual
                total_count += batch_size_actual

            preds_np = preds.to(torch.uint8).cpu().numpy()

            for idx, (pred, image_id) in enumerate(zip(preds_np, batch_image_ids)):
                binary_mask = pred.squeeze(0).astype(np.uint8)

                # Kaggle evaluates masks in each test image's original resolution.
                orig_h = int(orig_sizes[idx, 0].item())
                orig_w = int(orig_sizes[idx, 1].item())
                mask_img = Image.fromarray((binary_mask * 255).astype(np.uint8))
                mask_img = mask_img.resize((orig_w, orig_h), resample=NEAREST_RESAMPLE)
                binary_mask_for_submit = (np.array(mask_img) > 127).astype(np.uint8)

                encoded_mask = rle_encode(binary_mask_for_submit)
                submissions.append((image_id, encoded_mask))

                if len(vis_buffer) < num_vis:
                    image_vis = vis_images[idx].detach().cpu()
                    pred_vis = torch.from_numpy(binary_mask)
                    target_vis = None
                    if gt_available:
                        target_vis = gt_masks[idx].detach().cpu().squeeze(0)

                    vis_buffer.append((image_id, image_vis, pred_vis, target_vis))

    issues = validate_submission_rows(submissions, image_ids)

    with open(submission_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "encoded_mask"])
        writer.writerows(submissions)

    print("=" * 60)
    print("Inference complete")
    print(f"Model type: {model_type}")
    print(f"Model checkpoint: {model_path}")
    print(f"Test split file: {split_path}")
    print(f"Total test images: {len(image_ids)}")
    print(f"Submission saved to: {submission_path}")
    print(f"Visualization samples showed: {len(vis_buffer)}")
    print(f"Shape check: input {input_size} -> output {target_size} (PASSED)")

    if issues:
        print("Kaggle format check: FAILED")
        for issue in issues:
            print(f" - {issue}")
    else:
        print("Kaggle format check: PASSED")

    if gt_available and total_count > 0:
        mean_dice = total_dice / total_count
        print(f"Simulated Kaggle Dice score (test-set average): {mean_dice:.6f}")
    else:
        print("Simulated Kaggle Dice score: skipped (ground-truth masks unavailable)")
    print("=" * 60)

    if vis_buffer:
        visualize_predictions_grid(vis_buffer)


def build_argparser():
    parser = argparse.ArgumentParser(description="Oxford-IIIT Pet inference for Kaggle")
    parser.add_argument(
        "--model-type",
        type=str,
        default="UNet",
        choices=["UNet", "ResNet34_UNet"],
        help="Model architecture hint (ignored if --model-path filename clearly indicates model type)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="",
        help="Checkpoint path (.pth). If empty, auto-detect from saved_models/",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="dataset/oxford-iiit-pet",
        help="Path to Oxford-IIIT Pet dataset root",
    )
    parser.add_argument(
        "--hf-dataset-name",
        type=str,
        default="",
        help="HF dataset repo name, e.g. user/oxford-pet-nycu-lab2. If set, use HF dataset instead of --data-dir.",
    )
    parser.add_argument(
        "--hf-split",
        type=str,
        default="",
        help="HF split name to run inference on. If empty, auto-select by model type.",
    )
    parser.add_argument(
        "--split-file",
        type=str,
        default="",
        help="Optional custom test split file path",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Binary threshold for sigmoid outputs",
    )
    parser.add_argument(
        "--submission-path",
        type=str,
        default="submission.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--vis-dir",
        type=str,
        default="inference_outputs/vis_samples",
        help="Directory to save visualization examples",
    )
    parser.add_argument(
        "--num-vis",
        type=int,
        default=4,
        help="Number of test samples to visualize",
    )
    return parser


if __name__ == "__main__":
    arg_parser = build_argparser()
    run_inference(arg_parser.parse_args())
