#!/bin/bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
VENV_PATH="${VENV_PATH:-}"
DATASET_PATH="${DATASET_PATH:-/path/to/t5-midi/dataset}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_DIR/results/velocity-transformer}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
INIT_MODEL="${INIT_MODEL:-}"
MANIFEST_DIR="${MANIFEST_DIR:-$OUTPUT_DIR/manifests}"

D_MODEL="${D_MODEL:-384}"
NUM_LAYERS="${NUM_LAYERS:-8}"
NUM_HEADS="${NUM_HEADS:-8}"
D_FF="${D_FF:-1536}"
MAX_SEQUENCE_LENGTH="${MAX_SEQUENCE_LENGTH:-1024}"

PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-16}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-32}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-10}"
LEARNING_RATE="${LEARNING_RATE:-3e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
PRECISION="${PRECISION:-auto}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LOGGING_STEPS="${LOGGING_STEPS:-50}"
EVAL_EVERY_STEPS="${EVAL_EVERY_STEPS:-500}"
SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS:-500}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-0}"
ENABLE_GRADIENT_CHECKPOINTING="${ENABLE_GRADIENT_CHECKPOINTING:-0}"
ENABLE_COMPILE="${ENABLE_COMPILE:-0}"
DISABLE_MANIFEST_CACHE="${DISABLE_MANIFEST_CACHE:-0}"
TRAIN_MAX_SHARDS="${TRAIN_MAX_SHARDS:-0}"
VAL_MAX_SHARDS="${VAL_MAX_SHARDS:-0}"
TRAIN_SHARD_OFFSET="${TRAIN_SHARD_OFFSET:-0}"
VAL_SHARD_OFFSET="${VAL_SHARD_OFFSET:-0}"
TRAIN_SHARD_STRIDE="${TRAIN_SHARD_STRIDE:-1}"
VAL_SHARD_STRIDE="${VAL_SHARD_STRIDE:-1}"

if [[ -n "$VENV_PATH" ]]; then
  # shellcheck source=/dev/null
  source "$VENV_PATH/bin/activate"
fi

cd "$REPO_DIR"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
fi

cmd=(
  python train_velocity.py
  --dataset_path "$DATASET_PATH"
  --output_dir "$OUTPUT_DIR"
  --manifest_dir "$MANIFEST_DIR"
  --train_max_shards "$TRAIN_MAX_SHARDS"
  --val_max_shards "$VAL_MAX_SHARDS"
  --train_shard_offset "$TRAIN_SHARD_OFFSET"
  --val_shard_offset "$VAL_SHARD_OFFSET"
  --train_shard_stride "$TRAIN_SHARD_STRIDE"
  --val_shard_stride "$VAL_SHARD_STRIDE"
  --d_model "$D_MODEL"
  --num_layers "$NUM_LAYERS"
  --num_heads "$NUM_HEADS"
  --d_ff "$D_FF"
  --max_sequence_length "$MAX_SEQUENCE_LENGTH"
  --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE"
  --per_device_eval_batch_size "$PER_DEVICE_EVAL_BATCH_SIZE"
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
  --num_train_epochs "$NUM_TRAIN_EPOCHS"
  --learning_rate "$LEARNING_RATE"
  --weight_decay "$WEIGHT_DECAY"
  --warmup_ratio "$WARMUP_RATIO"
  --precision "$PRECISION"
  --num_workers "$NUM_WORKERS"
  --logging_steps "$LOGGING_STEPS"
  --eval_every_steps "$EVAL_EVERY_STEPS"
  --save_every_steps "$SAVE_EVERY_STEPS"
  --save_total_limit "$SAVE_TOTAL_LIMIT"
  --early_stopping_patience "$EARLY_STOPPING_PATIENCE"
)

if [[ -n "$RESUME_CHECKPOINT" ]]; then
  cmd+=(--resume_checkpoint "$RESUME_CHECKPOINT")
fi

if [[ -n "$INIT_MODEL" ]]; then
  cmd+=(--init_model "$INIT_MODEL")
fi

if [[ "$ENABLE_GRADIENT_CHECKPOINTING" == "1" ]]; then
  cmd+=(--gradient_checkpointing)
fi

if [[ "$ENABLE_COMPILE" == "1" ]]; then
  cmd+=(--compile)
fi

if [[ "$DISABLE_MANIFEST_CACHE" == "1" ]]; then
  cmd+=(--disable_manifest_cache)
fi

printf 'Running command:\n'
printf '  %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"
