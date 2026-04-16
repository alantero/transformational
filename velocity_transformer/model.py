from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .data_utils import IGNORE_INDEX
from .vocab import pad_token, velocity_events, vocab_size


@dataclass
class VelocityTransformerConfig:
    vocab_size: int = vocab_size
    pad_token_id: int = pad_token
    num_velocity_bins: int = velocity_events
    d_model: int = 384
    num_layers: int = 8
    num_heads: int = 8
    d_ff: int = 1536
    dropout: float = 0.1
    attention_dropout: float = 0.0
    activation_dropout: float = 0.0
    max_sequence_length: int = 1024
    num_relative_attention_buckets: int = 32
    relative_attention_max_distance: int = 1024
    label_smoothing: float = 0.0
    ordinal_loss_weight: float = 0.0

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, int | float]) -> "VelocityTransformerConfig":
        # Filter to known fields so old checkpoints (without new fields) and
        # future checkpoints (with unknown fields) both load cleanly.
        import dataclasses as _dc
        known = {f.name for f in _dc.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states


class SwiGLUFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.w_in = nn.Linear(d_model, d_ff * 2, bias=False)
        self.w_out = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate, value = self.w_in(hidden_states).chunk(2, dim=-1)
        hidden_states = F.silu(gate) * value
        hidden_states = self.dropout(hidden_states)
        return self.w_out(hidden_states)


class RelativePositionBias(nn.Module):
    def __init__(self, num_buckets: int, max_distance: int, num_heads: int) -> None:
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.num_heads = num_heads
        self.embedding = nn.Embedding(num_buckets, num_heads)

    def _relative_position_bucket(self, relative_position: torch.Tensor) -> torch.Tensor:
        num_buckets = self.num_buckets
        max_distance = self.max_distance

        half_buckets = num_buckets // 2
        sign_bucket = (relative_position > 0).to(torch.long) * half_buckets
        distance = relative_position.abs()

        max_exact = half_buckets // 2
        is_small = distance < max_exact

        clipped_distance = torch.clamp(distance, min=1)
        log_ratio = torch.log(clipped_distance.float() / max_exact) / math.log(max_distance / max_exact)
        large_position = max_exact + (log_ratio * (half_buckets - max_exact)).to(torch.long)
        large_position = torch.clamp(large_position, max=half_buckets - 1)
        bucket = torch.where(is_small, distance, large_position)
        return bucket + sign_bucket

    def forward(self, query_length: int, key_length: int, *, device: torch.device) -> torch.Tensor:
        context_position = torch.arange(query_length, device=device)[:, None]
        memory_position = torch.arange(key_length, device=device)[None, :]
        relative_position = memory_position - context_position
        bucket = self._relative_position_bucket(relative_position)
        values = self.embedding(bucket)
        return values.permute(2, 0, 1).unsqueeze(0)


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float,
        num_relative_attention_buckets: int,
        relative_attention_max_distance: int,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by num_heads={num_heads}")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = dropout

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.relative_position_bias = RelativePositionBias(
            num_buckets=num_relative_attention_buckets,
            max_distance=relative_attention_max_distance,
            num_heads=num_heads,
        )

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        query = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_bias = self.relative_position_bias(seq_len, seq_len, device=hidden_states.device).to(query.dtype)
        if attention_mask is not None:
            mask = attention_mask[:, None, None, :].to(query.dtype)
            neg_large = -1e4 if query.dtype in (torch.float16, torch.bfloat16) else -1e9
            attn_bias = attn_bias + (1.0 - mask) * neg_large

        attn_output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_bias,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.out_proj(attn_output)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, config: VelocityTransformerConfig) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(config.d_model)
        self.attention = MultiHeadSelfAttention(
            d_model=config.d_model,
            num_heads=config.num_heads,
            dropout=config.attention_dropout,
            num_relative_attention_buckets=config.num_relative_attention_buckets,
            relative_attention_max_distance=config.relative_attention_max_distance,
        )
        self.dropout = nn.Dropout(config.dropout)
        self.ffn_norm = RMSNorm(config.d_model)
        self.ffn = SwiGLUFeedForward(config.d_model, config.d_ff, dropout=config.activation_dropout)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        hidden_states = hidden_states + self.dropout(self.attention(self.attention_norm(hidden_states), attention_mask))
        hidden_states = hidden_states + self.dropout(self.ffn(self.ffn_norm(hidden_states)))
        return hidden_states


class VelocityTransformer(nn.Module):
    def __init__(self, config: VelocityTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.embeddings = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_token_id)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerEncoderBlock(config) for _ in range(config.num_layers)])
        self.final_norm = RMSNorm(config.d_model)
        self.classifier = nn.Linear(config.d_model, config.num_velocity_bins, bias=True)
        self.gradient_checkpointing = False

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    def enable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = True

    def disable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        hidden_states = self.embeddings(input_ids)
        hidden_states = self.embedding_dropout(hidden_states)

        if attention_mask is None:
            attention_mask = input_ids.ne(self.config.pad_token_id).long()

        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                hidden_states = checkpoint(block, hidden_states, attention_mask, use_reentrant=False)
            else:
                hidden_states = block(hidden_states, attention_mask)

        hidden_states = self.final_norm(hidden_states)
        logits = self.classifier(hidden_states)

        output = {"logits": logits}
        if labels is not None:
            valid_positions = labels.ne(IGNORE_INDEX)
            if valid_positions.any():
                valid_logits = logits[valid_positions].float()
                valid_labels = labels[valid_positions]

                ce_loss = F.cross_entropy(
                    valid_logits,
                    valid_labels,
                    label_smoothing=self.config.label_smoothing,
                )

                if self.config.ordinal_loss_weight > 0.0:
                    # Expected-value regression: the 32 velocity bins are ordinal
                    # (bin 20 is closer to bin 21 than to bin 0).  CE treats them
                    # as unordered categories, so predicting bin 20 when truth is
                    # 21 gets the same penalty as predicting bin 0.  The ordinal
                    # term E[bin] = Σ(i · softmax(logit_i)) and an L1 penalty
                    # against the true bin pushes the model toward "close" when
                    # it can't be exact.
                    probs = F.softmax(valid_logits, dim=-1)
                    bin_indices = torch.arange(
                        self.config.num_velocity_bins,
                        device=valid_logits.device,
                        dtype=torch.float32,
                    )
                    expected_bins = (probs * bin_indices).sum(dim=-1)
                    ordinal_loss = F.smooth_l1_loss(
                        expected_bins, valid_labels.float()
                    )
                    loss = ce_loss + self.config.ordinal_loss_weight * ordinal_loss
                    output["ce_loss"] = ce_loss
                    output["ordinal_loss"] = ordinal_loss
                else:
                    loss = ce_loss
            else:
                loss = logits.sum() * 0.0
            output["loss"] = loss
        return output

    def save_pretrained(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(output_dir, "model.pt"))
        with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as handle:
            json.dump(self.config.to_dict(), handle, indent=2, sort_keys=True)

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str,
        *,
        map_location: str | torch.device | None = None,
    ) -> "VelocityTransformer":
        config_path = os.path.join(model_dir, "config.json")
        weights_path = os.path.join(model_dir, "model.pt")
        with open(config_path, "r", encoding="utf-8") as handle:
            config = VelocityTransformerConfig.from_dict(json.load(handle))
        model = cls(config)
        state_dict = torch.load(weights_path, map_location=map_location or "cpu")
        model.load_state_dict(state_dict)
        return model
