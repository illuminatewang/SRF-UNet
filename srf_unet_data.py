"""Data loading, patch inference, loss, and metrics for SRF-UNet."""

import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from skimage.morphology import skeletonize
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


IMAGE_EXTENSIONS = {
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
}
IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)


def natural_key(path: Path) -> List[object]:
    """Sort filenames with embedded numbers in human order."""

    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def canonical_stem(path: Path) -> str:
    """Remove common mask suffixes before pairing files."""

    stem = path.stem
    lowered = stem.lower()
    for suffix in ("_gt", "_mask", "_manual1", "-gt", "-mask"):
        if lowered.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def list_images(folder: Path) -> List[Path]:
    """List supported image files in a directory."""

    if not folder.is_dir():
        raise FileNotFoundError(f"Directory not found: {folder}")
    return sorted(
        [
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ],
        key=natural_key,
    )


def build_pairs(image_dir: Path, mask_dir: Path) -> List[Tuple[Path, Path]]:
    """Pair image and mask files by normalized stem or sorted order."""

    images = list_images(image_dir)
    masks = list_images(mask_dir)
    mask_by_stem = {canonical_stem(mask): mask for mask in masks}

    if images and all(canonical_stem(image) in mask_by_stem for image in images):
        pairs = [
            (image, mask_by_stem[canonical_stem(image)]) for image in images
        ]
    elif len(images) == len(masks):
        pairs = list(zip(images, masks))
    else:
        raise ValueError(
            f"Cannot pair {len(images)} images in {image_dir} with "
            f"{len(masks)} masks in {mask_dir}."
        )

    if not pairs:
        raise ValueError(f"No image-mask pairs found in {image_dir}.")
    return pairs


def resolve_dataset_root(data_root: str, dataset: str) -> Path:
    """Resolve either a dataset collection root or a direct dataset path."""

    root = Path(data_root).expanduser().resolve()
    candidate = root / dataset
    return candidate if candidate.is_dir() else root


def resolve_split_dirs(dataset_root: Path, split: str) -> Tuple[Path, Path]:
    """Find image and mask directories for a dataset split."""

    split_dir = dataset_root / split
    candidates = (
        ("image", "mask"),
        ("images", "masks"),
        ("image", "masks"),
        ("images", "mask"),
        ("Original", "Ground truth"),
        ("image", "label"),
    )
    for image_name, mask_name in candidates:
        image_dir = split_dir / image_name
        mask_dir = split_dir / mask_name
        if image_dir.is_dir() and mask_dir.is_dir():
            return image_dir, mask_dir
    raise FileNotFoundError(
        f"Cannot find image and mask folders below {split_dir}."
    )


def read_image(path: Path) -> np.ndarray:
    """Read one color image in RGB order."""

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def read_mask(path: Path) -> np.ndarray:
    """Read one binary mask as an unsigned byte array."""

    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Cannot read mask: {path}")
    return (mask >= 20).astype(np.uint8)


def pad_to_patch(
    array: np.ndarray,
    patch_size: int,
    is_mask: bool = False,
) -> np.ndarray:
    """Pad only when an image is smaller than the requested patch."""

    height, width = array.shape[:2]
    pad_height = max(0, patch_size - height)
    pad_width = max(0, patch_size - width)
    if pad_height == 0 and pad_width == 0:
        return array
    border = cv2.BORDER_CONSTANT if is_mask else cv2.BORDER_REFLECT_101
    return cv2.copyMakeBorder(
        array,
        0,
        pad_height,
        0,
        pad_width,
        border,
        value=0,
    )


def sliding_positions(length: int, patch_size: int, stride: int) -> List[int]:
    """Return positions that cover an axis and always include its final edge."""

    if patch_size <= 0 or stride <= 0:
        raise ValueError("patch_size and stride must be positive.")
    if stride > patch_size:
        raise ValueError("stride must not exceed patch_size.")
    if length <= patch_size:
        return [0]
    positions = list(range(0, length - patch_size + 1, stride))
    final_position = length - patch_size
    if positions[-1] != final_position:
        positions.append(final_position)
    return positions


def normalize_image(image: np.ndarray) -> Tensor:
    """Apply the normalization used by the released training pipeline."""

    image = image.astype(np.float32) / 255.0
    image = (image - IMAGENET_MEAN) / IMAGENET_STD
    image = np.transpose(image, (2, 0, 1))
    return torch.from_numpy(np.ascontiguousarray(image)).float()


class RetinalPatchDataset(Dataset):
    """Generate reusable sliding-window training patches online."""

    def __init__(
        self,
        image_dir: Path,
        mask_dir: Path,
        patch_size: int = 224,
        stride: int = 112,
        augmentation: bool = True,
        cache_images: bool = True,
    ) -> None:
        self.pairs = build_pairs(image_dir, mask_dir)
        self.patch_size = int(patch_size)
        self.stride = int(stride)
        self.cache_images = cache_images
        self.cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        self.patch_index: List[Tuple[int, int, int]] = []

        for sample_index, (image_path, mask_path) in enumerate(self.pairs):
            image = read_image(image_path)
            mask = read_mask(mask_path)
            if image.shape[:2] != mask.shape[:2]:
                raise ValueError(
                    f"Image-mask size mismatch for {image_path.name}: "
                    f"{image.shape[:2]} versus {mask.shape[:2]}."
                )
            height, width = mask.shape
            for y in sliding_positions(
                max(height, self.patch_size),
                self.patch_size,
                self.stride,
            ):
                for x in sliding_positions(
                    max(width, self.patch_size),
                    self.patch_size,
                    self.stride,
                ):
                    self.patch_index.append((sample_index, y, x))

        transforms: List[A.BasicTransform] = []
        if augmentation:
            transforms.extend(
                (
                    A.Rotate(limit=90, p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.HorizontalFlip(p=0.5),
                )
            )
        self.transform = A.Compose(transforms)

    @property
    def image_count(self) -> int:
        return len(self.pairs)

    def __len__(self) -> int:
        return len(self.patch_index)

    def _load_pair(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        if index in self.cache:
            return self.cache[index]
        image_path, mask_path = self.pairs[index]
        image = pad_to_patch(read_image(image_path), self.patch_size)
        mask = pad_to_patch(
            read_mask(mask_path),
            self.patch_size,
            is_mask=True,
        )
        if self.cache_images:
            self.cache[index] = (image, mask)
        return image, mask

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        sample_index, y, x = self.patch_index[index]
        image, mask = self._load_pair(sample_index)
        size = self.patch_size
        augmented = self.transform(
            image=image[y : y + size, x : x + size],
            mask=mask[y : y + size, x : x + size],
        )
        image_tensor = normalize_image(augmented["image"])
        mask_array = np.ascontiguousarray(augmented["mask"].astype(np.float32))
        mask_tensor = torch.from_numpy(mask_array).unsqueeze(0)
        return image_tensor, mask_tensor


def create_training_loader(
    dataset_root: Path,
    split: str,
    batch_size: int,
    patch_size: int,
    stride: int,
    augmentation: bool,
    num_workers: int,
    persistent_workers: bool,
    worker_init_fn=None,
    generator=None,
) -> DataLoader:
    """Build the training patch loader."""

    image_dir, mask_dir = resolve_split_dirs(dataset_root, split)
    dataset = RetinalPatchDataset(
        image_dir,
        mask_dir,
        patch_size=patch_size,
        stride=stride,
        augmentation=augmentation,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=bool(persistent_workers and num_workers > 0),
        worker_init_fn=worker_init_fn,
        generator=generator,
    )


def structure_loss(logits: Tensor, masks: Tensor) -> Tensor:
    """Compute the region-aware weighted BCE and weighted IoU objective."""

    weights = 1 + 5 * torch.abs(
        F.avg_pool2d(masks, kernel_size=31, stride=1, padding=15) - masks
    )
    weighted_bce = F.binary_cross_entropy_with_logits(
        logits,
        masks,
        reduction="none",
    )
    weighted_bce = (weights * weighted_bce).sum((2, 3)) / weights.sum((2, 3))

    probabilities = torch.sigmoid(logits)
    intersection = (probabilities * masks * weights).sum((2, 3))
    union = ((probabilities + masks) * weights).sum((2, 3))
    weighted_iou = 1 - (intersection + 1) / (union - intersection + 1)
    return (weighted_bce + weighted_iou).mean()


def primary_output(outputs) -> Tensor:
    """Return raw logits from the released model output interface."""

    return outputs[0] if isinstance(outputs, (list, tuple)) else outputs


def gaussian_weight(patch_size: int, device: torch.device) -> Tensor:
    """Create a two-dimensional Gaussian blending map."""

    axis = torch.linspace(-1.0, 1.0, patch_size, device=device)
    y_axis, x_axis = torch.meshgrid(axis, axis, indexing="ij")
    weight = torch.exp(
        -0.5 * (x_axis.square() + y_axis.square()) / (0.5**2)
    )
    return weight.clamp_min(1e-3)


def sliding_window_predict(
    model: torch.nn.Module,
    image: np.ndarray,
    device: torch.device,
    patch_size: int,
    stride: int,
    patch_batch_size: int,
) -> Tensor:
    """Predict one full-resolution image with Gaussian patch blending."""

    original_height, original_width = image.shape[:2]
    padded = pad_to_patch(image, patch_size)
    padded_height, padded_width = padded.shape[:2]
    coordinates = [
        (y, x)
        for y in sliding_positions(padded_height, patch_size, stride)
        for x in sliding_positions(padded_width, patch_size, stride)
    ]
    probability_sum = torch.zeros(
        (padded_height, padded_width),
        dtype=torch.float32,
        device=device,
    )
    weight_sum = torch.zeros_like(probability_sum)
    weight = gaussian_weight(patch_size, device)

    model.eval()
    with torch.no_grad():
        for start in range(0, len(coordinates), patch_batch_size):
            batch_coordinates = coordinates[start : start + patch_batch_size]
            patches = [
                normalize_image(
                    padded[y : y + patch_size, x : x + patch_size]
                )
                for y, x in batch_coordinates
            ]
            batch = torch.stack(patches).to(device, non_blocking=True)
            logits = primary_output(model(batch))
            if logits.shape[-2:] != (patch_size, patch_size):
                logits = F.interpolate(
                    logits,
                    size=(patch_size, patch_size),
                    mode="bilinear",
                    align_corners=False,
                )
            probabilities = torch.sigmoid(logits[:, 0])
            for index, (y, x) in enumerate(batch_coordinates):
                probability_sum[
                    y : y + patch_size,
                    x : x + patch_size,
                ] += probabilities[index] * weight
                weight_sum[
                    y : y + patch_size,
                    x : x + patch_size,
                ] += weight

    probability = probability_sum / weight_sum.clamp_min(1e-8)
    return probability[:original_height, :original_width].cpu()


def binary_metrics(
    probability: np.ndarray,
    target: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    """Compute regional, classification, AUC, and centerline metrics."""

    prediction = probability >= threshold
    target_binary = target.astype(bool)
    true_positive = float(np.logical_and(prediction, target_binary).sum())
    true_negative = float(
        np.logical_and(~prediction, ~target_binary).sum()
    )
    false_positive = float(np.logical_and(prediction, ~target_binary).sum())
    false_negative = float(np.logical_and(~prediction, target_binary).sum())
    epsilon = 1e-8

    dice = (2 * true_positive + epsilon) / (
        2 * true_positive + false_positive + false_negative + epsilon
    )
    iou = (true_positive + epsilon) / (
        true_positive + false_positive + false_negative + epsilon
    )
    accuracy = (true_positive + true_negative) / target_binary.size
    sensitivity = true_positive / (true_positive + false_negative + epsilon)
    specificity = true_negative / (true_negative + false_positive + epsilon)
    precision = true_positive / (true_positive + false_positive + epsilon)

    auc = float("nan")
    if np.unique(target_binary).size == 2:
        auc = float(
            roc_auc_score(target_binary.reshape(-1), probability.reshape(-1))
        )

    prediction_skeleton = skeletonize(prediction)
    target_skeleton = skeletonize(target_binary)
    topology_precision = float(
        np.logical_and(prediction_skeleton, target_binary).sum()
    ) / (float(prediction_skeleton.sum()) + epsilon)
    topology_sensitivity = float(
        np.logical_and(target_skeleton, prediction).sum()
    ) / (float(target_skeleton.sum()) + epsilon)
    cldice = (
        2
        * topology_precision
        * topology_sensitivity
        / (topology_precision + topology_sensitivity + epsilon)
    )

    return {
        "Dice": dice,
        "IoU": iou,
        "Accuracy": accuracy,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Precision": precision,
        "AUC": auc,
        "clDice": cldice,
    }


def evaluate_model(
    model: torch.nn.Module,
    dataset_root: Path,
    split: str,
    device: torch.device,
    patch_size: int = 224,
    stride: int = 112,
    patch_batch_size: int = 16,
    threshold: float = 0.5,
    prediction_dir: Optional[Path] = None,
    max_images: int = 0,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Evaluate a split and optionally save binary prediction masks."""

    image_dir, mask_dir = resolve_split_dirs(dataset_root, split)
    pairs = build_pairs(image_dir, mask_dir)
    if max_images > 0:
        pairs = pairs[:max_images]
    if prediction_dir is not None:
        prediction_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for image_path, mask_path in pairs:
        image = read_image(image_path)
        target = read_mask(mask_path)
        probability = sliding_window_predict(
            model,
            image,
            device,
            patch_size,
            stride,
            patch_batch_size,
        ).numpy()
        if probability.shape != target.shape:
            raise ValueError(
                f"Prediction-mask size mismatch for {image_path.name}: "
                f"{probability.shape} versus {target.shape}."
            )
        metrics = binary_metrics(probability, target, threshold)
        rows.append({"Name": image_path.name, **metrics})

        if prediction_dir is not None:
            output = ((probability >= threshold) * 255).astype(np.uint8)
            cv2.imwrite(
                str(prediction_dir / f"{image_path.stem}.png"),
                output,
            )

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(f"No samples were evaluated in {split}.")
    summary = {
        name: float(frame[name].mean())
        for name in frame.columns
        if name != "Name"
    }
    summary["NumImages"] = float(len(frame))
    return frame, summary


def seed_worker(worker_id: int) -> None:
    """Seed data-loader workers from the PyTorch worker seed."""

    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
