"""Evaluate an SRF-UNet checkpoint at original image resolution."""

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import torch

from srf_unet import MODEL_CHANNELS, create_model
from srf_unet_data import evaluate_model, resolve_dataset_root


def choose_device(requested: str) -> torch.device:
    """Resolve CUDA requests safely on CPU-only systems."""

    if requested.startswith("cuda") and torch.cuda.is_available():
        return torch.device(requested)
    return torch.device("cpu")


def remove_parallel_prefix(state_dict: Dict[str, torch.Tensor]):
    """Remove a DataParallel prefix when present."""

    return {
        key.removeprefix("module."): value for key, value in state_dict.items()
    }


def load_model(
    checkpoint_path: Path,
    device: torch.device,
    fallback_size: str,
    fallback_classes: int,
    fallback_stages: int,
) -> Tuple[torch.nn.Module, Dict[str, object]]:
    """Build the correct model preset and load its checkpoint."""

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        metadata = checkpoint
        state_dict = checkpoint["model_state"]
        model_size = str(checkpoint.get("model_size", fallback_size))
        num_classes = int(checkpoint.get("num_classes", fallback_classes))
        series_stages = int(
            checkpoint.get("series_stages", fallback_stages)
        )
    else:
        metadata = {}
        state_dict = checkpoint
        model_size = fallback_size
        num_classes = fallback_classes
        series_stages = fallback_stages

    model = create_model(
        size=model_size,
        num_classes=num_classes,
        series_stages=series_stages,
    ).to(device)
    model.load_state_dict(remove_parallel_prefix(state_dict), strict=True)
    model.eval()
    metadata = {
        **metadata,
        "model_size": model_size,
        "num_classes": num_classes,
        "series_stages": series_stages,
    }
    return model, metadata


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line interface."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_root", default="./data")
    parser.add_argument("--dataset", default="DRIVE")
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--model_size",
        choices=tuple(MODEL_CHANNELS),
        default="base",
    )
    parser.add_argument("--num_classes", type=int, default=1)
    parser.add_argument("--series_stages", type=int, default=3)
    parser.add_argument("--patch_size", type=int, default=224)
    parser.add_argument("--infer_stride", type=int, default=112)
    parser.add_argument("--patch_batch_size", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", default="./test_results")
    parser.add_argument("--no_save_predictions", action="store_true")
    return parser


def main() -> None:
    """Load a checkpoint, evaluate it, and save a reproducible report."""

    arguments = build_parser().parse_args()
    checkpoint_path = Path(arguments.checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if arguments.num_classes != 1:
        raise ValueError("The released testing pipeline is binary and requires num_classes=1.")
    if arguments.patch_size <= 0 or arguments.patch_size % 32:
        raise ValueError("patch_size must be a positive multiple of 32.")
    if not 0 < arguments.infer_stride <= arguments.patch_size:
        raise ValueError("infer_stride must be in the range [1, patch_size].")

    device = choose_device(arguments.device)
    dataset_root = resolve_dataset_root(
        arguments.data_root,
        arguments.dataset,
    )
    output_dir = Path(arguments.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir = (
        None
        if arguments.no_save_predictions
        else output_dir / "predictions"
    )

    model, metadata = load_model(
        checkpoint_path,
        device,
        arguments.model_size,
        arguments.num_classes,
        arguments.series_stages,
    )
    details, summary = evaluate_model(
        model=model,
        dataset_root=dataset_root,
        split=arguments.split,
        device=device,
        patch_size=arguments.patch_size,
        stride=arguments.infer_stride,
        patch_batch_size=arguments.patch_batch_size,
        threshold=arguments.threshold,
        prediction_dir=prediction_dir,
        max_images=arguments.max_images,
    )
    details.to_excel(output_dir / "per_image_results.xlsx", index=False)
    pd.DataFrame([summary]).to_excel(
        output_dir / "summary.xlsx",
        index=False,
    )
    report = {
        "checkpoint": str(checkpoint_path),
        "dataset": arguments.dataset,
        "split": arguments.split,
        "device": str(device),
        "model_size": metadata["model_size"],
        "series_stages": metadata["series_stages"],
        **summary,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(
        f"Checkpoint: {checkpoint_path}\n"
        f"Dataset: {arguments.dataset}/{arguments.split}\n"
        f"Model size: {metadata['model_size']}\n"
        f"Dice: {summary['Dice']:.4f} | "
        f"clDice: {summary['clDice']:.4f} | "
        f"IoU: {summary['IoU']:.4f} | "
        f"Accuracy: {summary['Accuracy']:.4f}\n"
        f"Results: {output_dir}"
    )


if __name__ == "__main__":
    main()
