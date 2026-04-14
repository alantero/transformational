# velocity-transformer

Encoder-only Transformer for predicting MIDI `set_velocity_*` tokens from the existing `t5-midi` event vocabulary and shard format.

## Why this design

- **Bidirectional encoder** instead of GPT-style causal decoding.
- **Predict on `note_on` positions** after stripping explicit velocity tokens from the input.
- **Reuse the current `t5-midi` shards directly**: no dataset regeneration is required.
- **Keep it local-friendly**: the default model is small enough to train and run on a single T4.

The model uses:

- `RMSNorm`
- `SwiGLU`
- relative position bias inspired by T5
- PyTorch `scaled_dot_product_attention`, so CUDA builds can automatically use the best available kernel

## Expected dataset format

This repo expects the same shard layout already produced by `../t5-midi/data_processing/preprocessing2.py`:

```text
dataset_root/
  train/
    train_shard_000.pt
    train_shard_001.pt
  val/
    val_shard_000.pt
    val_shard_001.pt
```

Each shard must contain a 2-D padded tensor of token ids from the `t5-midi` vocabulary.

## Training locally

Recommended starting point:

```bash
python train_velocity.py \
  --dataset_path /path/to/dataset_root \
  --output_dir results/velocity-base \
  --d_model 384 \
  --num_layers 8 \
  --num_heads 8 \
  --d_ff 1536 \
  --max_sequence_length 1024 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 32 \
  --learning_rate 3e-4 \
  --num_train_epochs 10 \
  --precision auto \
  --eval_every_steps 500 \
  --save_every_steps 500
```

### EOS / Condor note

By default, the trainer now uses a **safe shard manifest cache** to speed up repeated starts on EOS:

- if a valid manifest exists, it is reused
- if it is missing or stale, the code falls back automatically to the original `t5-midi` behavior and reindexes shards by loading them
- if the manifest cannot be written, training still continues normally

This means the optimization is conservative: it does **not** replace the original shard-loading flow, it only avoids repeating it when possible.

You can disable this and force the old behavior with:

```bash
--disable_manifest_cache
```

If you prefer not to write manifest files into the dataset directory, point them somewhere else:

```bash
--manifest_dir results/velocity-base/manifests
```

For a smaller pilot:

```bash
python train_velocity.py \
  --dataset_path /path/to/dataset_root \
  --output_dir results/velocity-small \
  --d_model 256 \
  --num_layers 6 \
  --num_heads 8 \
  --d_ff 1024 \
  --per_device_train_batch_size 24 \
  --per_device_eval_batch_size 48
```

## Inference

From an existing shard sample:

```bash
python infer_velocity.py \
  --checkpoint_path results/velocity-base/best_model \
  --input_shard /path/to/dataset_root/val/val_shard_000.pt \
  --sample_index 0 \
  --output_tokens_path /tmp/predicted_sequence.pt
```

## Auditing data quality

To measure how expressive or flat the processed shards already are:

```bash
python audit_velocity_shards.py /path/to/dataset_root --split both --top_k 10
```

This reports:

- fraction of flat sequences
- fraction of sequences with `<= 2` velocity bins
- per-sequence velocity std and entropy summaries
- flattest and most expressive shards
- flattest and most expressive sequences

Important limitation:

- the current shard format does **not** preserve a direct mapping back to the original MIDI file, so this audit is sequence-level and shard-level, not truly per-source-file

From a MIDI file using `../t5-midi` for tokenization/rendering:

```bash
python infer_velocity.py \
  --checkpoint_path results/velocity-base/best_model \
  --midi_path /path/to/input.mid \
  --t5_midi_repo ../t5-midi \
  --output_tokens_path /tmp/predicted_sequence.pt \
  --output_midi_path /tmp/predicted_sequence.mid
```

## Condor

There is a wrapper in [job_examples/train_velocity_job_example.sh](/Users/agus/repositories/transformational/job_examples/train_velocity_job_example.sh:1) and a minimal submit example in [job_examples/train_velocity_example.sub](/Users/agus/repositories/transformational/job_examples/train_velocity_example.sub:1).

Typical usage:

1. Point `DATASET_PATH` to the existing shard directory.
2. Point `VENV_PATH` to your environment.
3. Keep the default `384/8/8/1536` model for the first serious run.
4. Lower the batch size or enable gradient checkpointing if memory gets tight on a T4.

## Outputs

Training writes:

- `run_config.json`
- `metrics.jsonl`
- rolling `checkpoint-*` directories with optimizer state
- `best_model/`
- `final_model/`

## Notes

- The dataset loader strips `set_velocity_*` tokens on the fly and turns the following `note_on` positions into labels.
- Existing shards from `t5-midi` are consumed directly, so this setup stays compatible with your current preprocessing pipeline and Condor workflow.
- If you want stricter per-instrument specialization later, you can train one checkpoint per instrument family on filtered shard subsets without changing the code.
