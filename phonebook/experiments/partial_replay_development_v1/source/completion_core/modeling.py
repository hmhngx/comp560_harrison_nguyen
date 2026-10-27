from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from completion_core.vocabulary import Vocabulary


@dataclass
class ModelConfig:
    seq_len: int
    d_model: int
    n_heads: int
    n_layers: int


class CompletionTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            enable_nested_tensor=False,
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        _b, t = idx.size()
        pos = torch.arange(t, dtype=torch.long, device=idx.device).unsqueeze(0)
        x = self.token_embedding(idx) + self.position_embedding(pos)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            t, device=idx.device
        )
        x = self.transformer(x, mask=causal_mask, is_causal=True)
        return self.lm_head(self.ln_f(x))


def _load_checkpoint_payload(weights_path: Path, device: str) -> dict[str, Any]:
    if not weights_path.exists():
        raise FileNotFoundError(f"Could not find model weights at {weights_path}")

    raw = torch.load(weights_path, map_location=device)
    if isinstance(raw, dict) and "state_dict" in raw:
        payload = raw
    elif isinstance(raw, dict):
        payload = {"state_dict": raw}
    else:
        raise TypeError(f"Unsupported checkpoint format in {weights_path}")

    return payload


def _infer_layer_count(state_dict: dict[str, torch.Tensor], fallback: int) -> int:
    layer_indices: set[int] = set()
    for key in state_dict.keys():
        if key.startswith("transformer.layers."):
            parts = key.split(".")
            if len(parts) >= 3 and parts[2].isdigit():
                layer_indices.add(int(parts[2]))
    if not layer_indices:
        return fallback
    return max(layer_indices) + 1


def load_model_and_config(
    weights_path: Path,
    vocab: Vocabulary,
    device: str,
    default_d_model: int = 128,
    default_n_heads: int = 4,
    default_n_layers: int = 4,
) -> tuple[CompletionTransformer, ModelConfig]:
    payload = _load_checkpoint_payload(weights_path, device)
    state_dict: dict[str, torch.Tensor] = payload["state_dict"]

    model_meta = payload.get("model_config", {})
    seq_len = state_dict["position_embedding.weight"].shape[0]
    d_model = int(model_meta.get("d_model", state_dict["token_embedding.weight"].shape[1]))
    n_heads = int(model_meta.get("n_heads", default_n_heads))
    n_layers = int(model_meta.get("n_layers", _infer_layer_count(state_dict, default_n_layers)))

    cfg = ModelConfig(
        seq_len=seq_len,
        d_model=d_model if d_model > 0 else default_d_model,
        n_heads=n_heads if n_heads > 0 else default_n_heads,
        n_layers=n_layers if n_layers > 0 else default_n_layers,
    )

    model = CompletionTransformer(
        vocab_size=vocab.vocab_size,
        seq_len=cfg.seq_len,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, cfg


def make_checkpoint_payload(model: nn.Module, cfg: ModelConfig) -> dict[str, Any]:
    return {
        "state_dict": model.state_dict(),
        "model_config": {
            "seq_len": cfg.seq_len,
            "d_model": cfg.d_model,
            "n_heads": cfg.n_heads,
            "n_layers": cfg.n_layers,
        },
    }
