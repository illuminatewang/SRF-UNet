"""Train SRF-UNet on retinal vessel segmentation datasets."""

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from srf_unet import MODEL_CHANNELS, create_model
from srf_unet_data import (
    create_training_loader,
    evaluate_model,
    primary_output,
    resolve_dataset_root,
    seed_worker,
    structure_loss,
)


def set_seed(seed: int, deterministic: bool) -> None:
    """Configure random generators for one independent run."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def choose_device(requested: str) -> torch.device:
    """Resolve CUDA requests safely on CPU-only systems."""

    if requested.startswith("cuda") and torch.cuda.is_available():
        return torch.device(requested)
    return torch.device("cpu")


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_dice: float,
    arguments: argparse.Namespace,
) -> None:
    """Save model weights with the architecture metadata required for testing."""

    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "best_dice": best_dice,
            "model_size": arguments.model_size,
            "num_classes": arguments.num_classes,
            "in_channels": 3,
            "series_stages": arguments.series_stages,
            "channels": MODEL_CHANNELS[arguments.model_size],
        },
        path,
    )


def train_one_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    gradient_clip: float,
    max_steps: int,
) -> Dict[str, float]:
    """Run one epoch and return its average loss and update count."""

    model.train()
    loss_sum = 0.0
    steps = 0
    patch_count = 0

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = primary_output(model(images))
        loss = structure_loss(logits, masks)
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), gradient_clip)
        optimizer.step()

        loss_sum += float(loss.detach())
        steps += 1
        patch_count += images.shape[0]
        if max_steps > 0 and steps >= max_steps:
            break

    return {
        "Loss": loss_sum / max(steps, 1),
        "Steps": float(steps),
        "Patches": float(patch_count),
    }


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line interface."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", default="./data")
    parser.add_argument("--dataset", default="DRIVE")
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--selection_split", default="test")
    parser.add_argument(
        "--model_size",
        choices=tuple(MODEL_CHANNELS),
        default="base",
    )
    parser.add_argument("--num_classes", type=int, default=1)
    parser.add_argument("--series_stages", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--patch_batch_size", type=int, default=16)
    parser.add_argument("--patch_size", type=int, default=224)
    parser.add_argument("--train_stride", type=int, default=112)
    parser.add_argument("--infer_stride", type=int, default=112)
    parser.add_argument("--learning_rate", type=float, default=0.0005)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--gradient_clip", type=float, default=0.5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--persistent_workers", action="store_true")
    parser.add_argument("--no_augmentation", action="store_true")
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--max_eval_images", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", default="./outputs")
    return parser


def validate_arguments(arguments: argparse.Namespace) -> None:
    """Reject incompatible training settings before starting a run."""

    if arguments.num_classes != 1:
        raise ValueError("The released training pipeline is binary and requires num_classes=1.")
    if arguments.patch_size <= 0 or arguments.patch_size % 32:
        raise ValueError("patch_size must be a positive multiple of 32.")
    if not 0 < arguments.train_stride <= arguments.patch_size:
        raise ValueError("train_stride must be in the range [1, patch_size].")
    if not 0 < arguments.infer_stride <= arguments.patch_size:
        raise ValueError("infer_stride must be in the range [1, patch_size].")
    if arguments.epochs <= 0 or arguments.runs <= 0:
        raise ValueError("epochs and runs must be positive.")
    if arguments.batch_size <= 0 or arguments.patch_batch_size <= 0:
        raise ValueError("batch sizes must be positive.")


def main() -> None:
    """Train one or more reproducible SRF-UNet runs."""

    arguments = build_parser().parse_args()
    validate_arguments(arguments)
    device = choose_device(arguments.device)
    dataset_root = resolve_dataset_root(
        arguments.data_root,
        arguments.dataset,
    )
    output_root = Path(arguments.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    for run_index in range(1, arguments.runs + 1):
        run_seed = arguments.seed + run_index - 1
        set_seed(run_seed, arguments.deterministic)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_id = (
            f"{arguments.dataset}_srf_unet_{arguments.model_size}_"
            f"r{run_index}_s{run_seed}_{timestamp}"
        )
        run_dir = output_root / run_id
        run_dir.mkdir(parents=True, exist_ok=False)

        generator = torch.Generator()
        generator.manual_seed(run_seed)
        loader = create_training_loader(
            dataset_root=dataset_root,
            split=arguments.train_split,
            batch_size=arguments.batch_size,
            patch_size=arguments.patch_size,
            stride=arguments.train_stride,
            augmentation=not arguments.no_augmentation,
            num_workers=arguments.num_workers,
            persistent_workers=arguments.persistent_workers,
            worker_init_fn=seed_worker,
            generator=generator,
        )
        model = create_model(
            size=arguments.model_size,
            num_classes=arguments.num_classes,
            series_stages=arguments.series_stages,
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=arguments.learning_rate,
            weight_decay=arguments.weight_decay,
        )
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=arguments.epochs,
            eta_min=1e-6,
        )

        configuration = vars(arguments).copy()
        configuration.update(
            {
                "run_id": run_id,
                "run_seed": run_seed,
                "device_used": str(device),
                "channels": MODEL_CHANNELS[arguments.model_size],
                "training_images": loader.dataset.image_count,
                "patches_per_epoch": len(loader.dataset),
            }
        )
        (run_dir / "config.json").write_text(
            json.dumps(configuration, indent=2),
            encoding="utf-8",
        )

        print(
            f"Run: {run_id}\n"
            f"Model: SRF-UNet-{arguments.model_size} | "
            f"Channels: {MODEL_CHANNELS[arguments.model_size]}\n"
            f"Dataset: {dataset_root} | Device: {device}\n"
            f"Training images: {loader.dataset.image_count} | "
            f"Patches per epoch: {len(loader.dataset)}"
        )

        best_dice = -1.0
        history = []
        start_time = time.perf_counter()
        for epoch in range(1, arguments.epochs + 1):
            epoch_start = time.perf_counter()
            training = train_one_epoch(
                model,
                loader,
                optimizer,
                device,
                arguments.gradient_clip,
                arguments.max_steps,
            )
            scheduler.step()
            row = {
                "Epoch": epoch,
                "LearningRate": optimizer.param_groups[0]["lr"],
                "TrainLoss": training["Loss"],
                "OptimizerSteps": training["Steps"],
                "Patches": training["Patches"],
                "EpochSeconds": time.perf_counter() - epoch_start,
            }

            should_evaluate = (
                arguments.eval_every > 0
                and (
                    epoch % arguments.eval_every == 0
                    or epoch == arguments.epochs
                )
            )
            if should_evaluate:
                _, summary = evaluate_model(
                    model=model,
                    dataset_root=dataset_root,
                    split=arguments.selection_split,
                    device=device,
                    patch_size=arguments.patch_size,
                    stride=arguments.infer_stride,
                    patch_batch_size=arguments.patch_batch_size,
                    threshold=arguments.threshold,
                    max_images=arguments.max_eval_images,
                )
                row.update(
                    {
                        f"Selection{name}": value
                        for name, value in summary.items()
                    }
                )
                if summary["Dice"] > best_dice:
                    best_dice = summary["Dice"]
                    save_checkpoint(
                        run_dir / "best.pth",
                        model,
                        optimizer,
                        epoch,
                        best_dice,
                        arguments,
                    )

            save_checkpoint(
                run_dir / "last.pth",
                model,
                optimizer,
                epoch,
                best_dice,
                arguments,
            )
            history.append(row)
            pd.DataFrame(history).to_excel(
                run_dir / "training_history.xlsx",
                index=False,
            )
            dice_text = (
                f", selection Dice={row['SelectionDice']:.4f}"
                if "SelectionDice" in row
                else ""
            )
            print(
                f"Epoch {epoch:03d}/{arguments.epochs:03d}: "
                f"loss={training['Loss']:.4f}{dice_text}"
            )

        elapsed = time.perf_counter() - start_time
        print(
            f"Finished {run_id} in {elapsed / 60:.2f} minutes. "
            f"Best selection Dice: {best_dice:.4f}"
        )


if __name__ == "__main__":
    main()
