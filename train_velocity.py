#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from velocity_transformer.data_utils import IGNORE_INDEX
from velocity_transformer.dataset import ShardedMIDIVelocityDataset, VelocityPredictionCollator
from velocity_transformer.model import VelocityTransformer, VelocityTransformerConfig
from velocity_transformer.training_utils import (
    autocast_context,
    create_cosine_schedule_with_warmup,
    detect_device,
    prune_checkpoints,
    resolve_autocast_dtype,
    save_json,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a lightweight encoder-only transformer to predict MIDI velocity bins."
    )
    parser.add_argument("--dataset_path", type=str, required=True, help="Base directory with train/ and val/ shards")
    parser.add_argument("--output_dir", type=str, default="results/velocity-transformer", help="Directory for checkpoints and logs")
    parser.add_argument("--resume_checkpoint", type=str, default=None, help="Checkpoint directory containing model.pt and training_state.pt")
    parser.add_argument("--init_model", type=str, default=None, help="Model directory to initialize weights from, without optimizer state")

    parser.add_argument("--d_model", type=int, default=384)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=1536)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--attention_dropout", type=float, default=0.0)
    parser.add_argument("--activation_dropout", type=float, default=0.0)
    parser.add_argument("--max_sequence_length", type=int, default=1024)
    parser.add_argument("--num_relative_attention_buckets", type=int, default=32)
    parser.add_argument("--relative_attention_max_distance", type=int, default=1024)
    parser.add_argument("--label_smoothing", type=float, default=0.0)

    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--max_train_steps", type=int, default=0, help="Override the total number of optimizer steps")
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--precision", choices=["auto", "fp16", "bf16", "none"], default="auto")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--compile", action="store_true", help="Compile the model with torch.compile when available")

    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", help="auto, cuda, cpu, mps, or explicit device string")
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--eval_every_steps", type=int, default=500, help="0 means eval only at epoch end")
    parser.add_argument("--save_every_steps", type=int, default=500, help="0 means save only at epoch end")
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--early_stopping_patience", type=int, default=0, help="Number of eval calls without improvement before stopping; 0 disables")
    parser.add_argument("--limit_train_batches", type=int, default=0, help="For debugging or smoke tests")
    parser.add_argument("--limit_val_batches", type=int, default=0, help="For debugging or smoke tests")

    parser.add_argument("--min_notes_per_sequence", type=int, default=1)
    parser.add_argument("--default_velocity_bin", type=int, default=None)
    return parser.parse_args()


def move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: tensor.to(device, non_blocking=True) for name, tensor in batch.items()}


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def build_model_config(args: argparse.Namespace) -> VelocityTransformerConfig:
    return VelocityTransformerConfig(
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
        attention_dropout=args.attention_dropout,
        activation_dropout=args.activation_dropout,
        max_sequence_length=args.max_sequence_length,
        num_relative_attention_buckets=args.num_relative_attention_buckets,
        relative_attention_max_distance=args.relative_attention_max_distance,
        label_smoothing=args.label_smoothing,
    )


def compute_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    valid_mask = labels.ne(IGNORE_INDEX)
    valid_count = int(valid_mask.sum().item())
    if valid_count == 0:
        return {
            "loss": 0.0,
            "accuracy": 0.0,
            "mae_bins": 0.0,
            "within_1_bin": 0.0,
            "supervised_notes": 0.0,
        }

    valid_logits = logits[valid_mask]
    valid_labels = labels[valid_mask]
    loss = F.cross_entropy(valid_logits, valid_labels, reduction="mean").item()
    predictions = valid_logits.argmax(dim=-1)
    abs_error = (predictions - valid_labels).abs()
    return {
        "loss": loss,
        "accuracy": predictions.eq(valid_labels).float().mean().item(),
        "mae_bins": abs_error.float().mean().item(),
        "within_1_bin": abs_error.le(1).float().mean().item(),
        "supervised_notes": float(valid_count),
    }


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    *,
    device: torch.device,
    autocast_dtype: torch.dtype | None,
    limit_batches: int = 0,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_abs_error = 0.0
    total_within_1 = 0
    total_notes = 0

    for batch_idx, batch in enumerate(dataloader):
        if limit_batches and batch_idx >= limit_batches:
            break

        batch = move_batch_to_device(batch, device)
        with autocast_context(device, autocast_dtype):
            outputs = model(**batch)

        logits = outputs["logits"]
        labels = batch["labels"]
        valid_mask = labels.ne(IGNORE_INDEX)
        valid_count = int(valid_mask.sum().item())
        if valid_count == 0:
            continue

        valid_logits = logits[valid_mask]
        valid_labels = labels[valid_mask]
        loss_sum = F.cross_entropy(valid_logits, valid_labels, reduction="sum").item()
        predictions = valid_logits.argmax(dim=-1)
        abs_error = (predictions - valid_labels).abs()

        total_loss += loss_sum
        total_correct += int(predictions.eq(valid_labels).sum().item())
        total_abs_error += float(abs_error.sum().item())
        total_within_1 += int(abs_error.le(1).sum().item())
        total_notes += valid_count

    model.train()
    if total_notes == 0:
        return {
            "eval_loss": 0.0,
            "eval_accuracy": 0.0,
            "eval_mae_bins": 0.0,
            "eval_within_1_bin": 0.0,
            "eval_supervised_notes": 0.0,
        }

    return {
        "eval_loss": total_loss / total_notes,
        "eval_accuracy": total_correct / total_notes,
        "eval_mae_bins": total_abs_error / total_notes,
        "eval_within_1_bin": total_within_1 / total_notes,
        "eval_supervised_notes": float(total_notes),
    }


def save_checkpoint_bundle(
    checkpoint_dir: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None,
    scaler: torch.amp.GradScaler | None,
    metadata: dict,
) -> None:
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model_to_save = unwrap_model(model)
    assert isinstance(model_to_save, VelocityTransformer)
    model_to_save.save_pretrained(str(checkpoint_dir))
    save_json(checkpoint_dir / "metadata.json", metadata)

    if optimizer is not None and scheduler is not None:
        torch.save(
            {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict() if scaler is not None else None,
                "metadata": metadata,
            },
            checkpoint_dir / "training_state.pt",
        )


def maybe_save_named_snapshot(output_dir: Path, name: str, *, model: torch.nn.Module, metadata: dict) -> None:
    snapshot_dir = output_dir / name
    if snapshot_dir.exists():
        shutil.rmtree(snapshot_dir)
    save_checkpoint_bundle(
        snapshot_dir,
        model=model,
        optimizer=None,
        scheduler=None,
        scaler=None,
        metadata=metadata,
    )


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "run_config.json", vars(args))

    device = detect_device(args.device)
    autocast_dtype = resolve_autocast_dtype(device, args.precision)
    use_grad_scaler = device.type == "cuda" and autocast_dtype == torch.float16

    print(f"Using device: {device}")
    print(f"Autocast dtype: {autocast_dtype}")

    train_dir = Path(args.dataset_path).expanduser().resolve() / "train"
    val_dir = Path(args.dataset_path).expanduser().resolve() / "val"
    train_dataset = ShardedMIDIVelocityDataset(
        str(train_dir),
        min_notes_per_sequence=args.min_notes_per_sequence,
        default_velocity_bin=args.default_velocity_bin,
    )
    val_dataset = ShardedMIDIVelocityDataset(
        str(val_dir),
        min_notes_per_sequence=args.min_notes_per_sequence,
        default_velocity_bin=args.default_velocity_bin,
    )
    collator = VelocityPredictionCollator()

    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=collator,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.per_device_eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        **loader_kwargs,
    )

    effective_train_batches = len(train_loader)
    if args.limit_train_batches:
        effective_train_batches = min(effective_train_batches, args.limit_train_batches)
    steps_per_epoch = max(1, math.ceil(effective_train_batches / args.gradient_accumulation_steps))
    total_steps = args.max_train_steps if args.max_train_steps > 0 else steps_per_epoch * args.num_train_epochs
    warmup_steps = args.warmup_steps if args.warmup_steps > 0 else int(total_steps * args.warmup_ratio)

    if args.resume_checkpoint:
        model = VelocityTransformer.from_pretrained(args.resume_checkpoint, map_location="cpu")
    elif args.init_model:
        model = VelocityTransformer.from_pretrained(args.init_model, map_location="cpu")
    else:
        model = VelocityTransformer(build_model_config(args))

    if args.gradient_checkpointing:
        model.enable_gradient_checkpointing()

    model.to(device)
    run_model: torch.nn.Module = model
    if args.compile:
        if not hasattr(torch, "compile"):
            raise RuntimeError("torch.compile is not available in this environment")
        run_model = torch.compile(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
    )
    scheduler = create_cosine_schedule_with_warmup(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_lr_ratio=args.min_lr_ratio,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_grad_scaler)

    start_epoch = 0
    global_step = 0
    best_eval_loss = float("inf")
    evals_without_improvement = 0

    if args.resume_checkpoint:
        training_state_path = Path(args.resume_checkpoint) / "training_state.pt"
        if not training_state_path.exists():
            raise FileNotFoundError(f"Missing training_state.pt in {args.resume_checkpoint}")
        state = torch.load(training_state_path, map_location="cpu")
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        move_optimizer_state(optimizer, device)
        if use_grad_scaler and state.get("scaler") is not None:
            scaler.load_state_dict(state["scaler"])
        metadata = state.get("metadata", {})
        start_epoch = int(metadata.get("epoch", 0))
        global_step = int(metadata.get("global_step", 0))
        best_eval_loss = float(metadata.get("best_eval_loss", float("inf")))
        evals_without_improvement = int(metadata.get("evals_without_improvement", 0))
        print(f"Resumed from {args.resume_checkpoint} at step {global_step}, epoch {start_epoch}")

    metrics_log_path = output_dir / "metrics.jsonl"

    def append_metrics(event: dict) -> None:
        with open(metrics_log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")

    def run_evaluation(epoch: int, step: int, reason: str) -> dict[str, float]:
        metrics = evaluate(
            run_model,
            val_loader,
            device=device,
            autocast_dtype=autocast_dtype,
            limit_batches=args.limit_val_batches,
        )
        event = {
            "type": "eval",
            "reason": reason,
            "epoch": epoch,
            "global_step": step,
            **metrics,
        }
        append_metrics(event)
        print(
            "[eval]",
            f"step={step}",
            f"loss={metrics['eval_loss']:.4f}",
            f"acc={metrics['eval_accuracy']:.4f}",
            f"mae={metrics['eval_mae_bins']:.4f}",
            f"within1={metrics['eval_within_1_bin']:.4f}",
        )
        return metrics

    def maybe_save_training_checkpoint(epoch: int, step: int, extra: dict[str, float], tag: str) -> None:
        checkpoint_dir = output_dir / f"checkpoint-{step:08d}"
        metadata = {
            "tag": tag,
            "epoch": epoch,
            "global_step": step,
            "best_eval_loss": best_eval_loss,
            "evals_without_improvement": evals_without_improvement,
            **extra,
        }
        save_checkpoint_bundle(
            checkpoint_dir,
            model=run_model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler if use_grad_scaler else None,
            metadata=metadata,
        )
        prune_checkpoints(str(output_dir), keep_last=args.save_total_limit)

    optimizer.zero_grad(set_to_none=True)
    training_start = time.time()
    stop_training = False

    for epoch in range(start_epoch, args.num_train_epochs):
        run_model.train()
        micro_loss_accumulator = 0.0
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(train_loader):
            if args.limit_train_batches and batch_idx >= args.limit_train_batches:
                break
            if global_step >= total_steps:
                stop_training = True
                break

            batch = move_batch_to_device(batch, device)
            with autocast_context(device, autocast_dtype):
                outputs = run_model(**batch)
                loss = outputs["loss"] / args.gradient_accumulation_steps

            if use_grad_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            micro_loss_accumulator += loss.item()

            is_update_step = (batch_idx + 1) % args.gradient_accumulation_steps == 0
            is_last_batch = batch_idx + 1 == effective_train_batches
            if not is_update_step and not is_last_batch:
                continue

            if use_grad_scaler:
                scaler.unscale_(optimizer)
            if args.max_grad_norm > 0:
                clip_grad_norm_(model.parameters(), args.max_grad_norm)

            if use_grad_scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

            if global_step % args.logging_steps == 0 or global_step == 1:
                batch_metrics = compute_metrics(outputs["logits"].detach(), batch["labels"])
                elapsed = time.time() - training_start
                event = {
                    "type": "train",
                    "epoch": epoch,
                    "global_step": global_step,
                    "train_loss": micro_loss_accumulator,
                    "train_batch_accuracy": batch_metrics["accuracy"],
                    "train_batch_mae_bins": batch_metrics["mae_bins"],
                    "train_batch_within_1_bin": batch_metrics["within_1_bin"],
                    "learning_rate": scheduler.get_last_lr()[0],
                    "elapsed_seconds": elapsed,
                }
                append_metrics(event)
                print(
                    "[train]",
                    f"step={global_step}/{total_steps}",
                    f"loss={micro_loss_accumulator:.4f}",
                    f"acc={batch_metrics['accuracy']:.4f}",
                    f"mae={batch_metrics['mae_bins']:.4f}",
                    f"lr={scheduler.get_last_lr()[0]:.6f}",
                )
                micro_loss_accumulator = 0.0

            if args.eval_every_steps and global_step % args.eval_every_steps == 0:
                metrics = run_evaluation(epoch, global_step, reason="steps")
                if metrics["eval_loss"] < best_eval_loss:
                    best_eval_loss = metrics["eval_loss"]
                    evals_without_improvement = 0
                    maybe_save_named_snapshot(
                        output_dir,
                        "best_model",
                        model=run_model,
                        metadata={
                            "epoch": epoch,
                            "global_step": global_step,
                            "best_eval_loss": best_eval_loss,
                            **metrics,
                        },
                    )
                else:
                    evals_without_improvement += 1

                if args.early_stopping_patience > 0 and evals_without_improvement >= args.early_stopping_patience:
                    print("Early stopping triggered")
                    stop_training = True
                    break

            if args.save_every_steps and global_step % args.save_every_steps == 0:
                maybe_save_training_checkpoint(epoch, global_step, extra={}, tag="steps")

            if global_step >= total_steps:
                stop_training = True
                break

        metrics = run_evaluation(epoch, global_step, reason="epoch_end")
        if metrics["eval_loss"] < best_eval_loss:
            best_eval_loss = metrics["eval_loss"]
            evals_without_improvement = 0
            maybe_save_named_snapshot(
                output_dir,
                "best_model",
                model=run_model,
                metadata={
                    "epoch": epoch,
                    "global_step": global_step,
                    "best_eval_loss": best_eval_loss,
                    **metrics,
                },
            )
        else:
            evals_without_improvement += 1

        maybe_save_training_checkpoint(epoch + 1, global_step, extra=metrics, tag="epoch_end")

        if args.early_stopping_patience > 0 and evals_without_improvement >= args.early_stopping_patience:
            print("Early stopping triggered after epoch evaluation")
            break

        if stop_training:
            break

    final_metadata = {
        "global_step": global_step,
        "best_eval_loss": best_eval_loss,
        "training_seconds": time.time() - training_start,
    }
    maybe_save_named_snapshot(output_dir, "final_model", model=run_model, metadata=final_metadata)
    print(f"Training finished. Best eval loss: {best_eval_loss:.4f}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
