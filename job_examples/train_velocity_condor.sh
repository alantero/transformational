#!/bin/bash
set -euo pipefail

source /eos/home-a/alantero/t5venv/bin/activate

echo -e "Tu job Condor\nClusterId: ${CLUSTER_ID}\nProcId: ${PROCESS_ID}\nHa comenzado en $(hostname) a las $(date)" \
  | mail -s "Job ${CLUSTER_ID}.${PROCESS_ID} started" agusem@hotmail.com

nvidia-smi

python /eos/home-a/alantero/transformational/train_velocity.py \
    --dataset_path "/eos/home-a/alantero/datasets/pretraining_velocity_shards" \
    --output_dir "/eos/home-a/alantero/transformational/results/velocity-base" \
    --manifest_dir "/eos/home-a/alantero/transformational/results/velocity-base/manifests" \
    --tensorboard always \
    --tensorboard_dir "/eos/home-a/alantero/transformational/results/velocity-base/tensorboard" \
    --d_model 384 \
    --num_layers 8 \
    --num_heads 8 \
    --d_ff 1536 \
    --max_sequence_length 1024 \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 128 \
    --gradient_accumulation_steps 2 \
    --num_train_epochs 10 \
    --learning_rate 3e-4 \
    --warmup_steps 1000 \
    --precision fp16 \
    --num_workers 32 \
    --logging_steps 500 \
    --eval_every_steps 2000 \
    --save_every_steps 500 \
    --save_total_limit 3 \
    --progress_bar never \
    --disable_manifest_cache

# Para abrir TensorBoard con las rutas de este ejemplo:
# source /eos/home-a/alantero/t5venv/bin/activate
# tensorboard --logdir /eos/home-a/alantero/transformational/results/velocity-base/tensorboard --port 6006
