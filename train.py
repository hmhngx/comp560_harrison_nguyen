"""
Train a GPT-style decoder-only transformer on character-level completion sequences.

Data format: each line is "<pad><input>=<output>\n", tokenised and stored as
uint16 token IDs in train.bin / val.bin alongside a meta.pkl vocabulary file.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
import time
import csv

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from completion_core.modeling import (
    CompletionTransformer,
    ModelConfig,
    make_checkpoint_payload,
)
from completion_core.training import SequenceDataset, evaluate_epoch_metrics, run_epoch
from completion_core.vocabulary import Vocabulary


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    data_dir: Path = Path("data")
    out_dir: Path = Path("out")
    adapt_from: Path | None = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Model
    embedding_dim: int = 128
    n_heads: int = 4
    n_layers: int = 4

    # Training
    batch_size: int = 32
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-1
    grad_clip: float = 1.0
    seed: int = 42
    log_interval: int = 50  # batches between progress prints
    accuracy_interval: int = 10  # epochs between no-grad train/val accuracy checks
    l2sp_lambda: float = 0.0  # L2-SP penalty strength; 0.0 means off


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_file", type=str, nargs="?", default=None,
                        help="Optional path to a nanoGPT-style python config file")
    # default=argparse.SUPPRESS (not a real default value) so an unset flag
    # leaves no attribute on `args` at all. hasattr(args, ...) below then
    # detects "was this flag actually given" correctly for every syntax
    # argparse accepts (`--flag value`, `--flag=value`, unambiguous
    # abbreviations) -- unlike the previous `"--flag" in sys.argv` check,
    # which only ever matched the exact `--flag value` token pair and
    # silently no-op'd for `=` syntax (verified: bypassed the --l2sp-lambda
    # negative-value rejection below).
    parser.add_argument("--data-dir", type=Path, default=argparse.SUPPRESS)
    parser.add_argument("--out-dir", type=Path, default=argparse.SUPPRESS)
    parser.add_argument("--device", type=str, default=argparse.SUPPRESS)
    parser.add_argument("--embedding-dim", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--n-heads", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--n-layers", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--batch-size", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--epochs", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--lr", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=argparse.SUPPRESS, help="Random seed (default: 42)")
    parser.add_argument(
        "--accuracy-interval",
        type=int,
        default=argparse.SUPPRESS,
        help="Run no-grad train/val token+sequence accuracy every N epochs (default: 10)",
    )
    parser.add_argument(
        "--adapt-from",
        type=Path,
        default=argparse.SUPPRESS,
        help="Path to a checkpoint to initialize model weights for model adaptation / fine-tuning (default: None)",
    )
    parser.add_argument(
        "--l2sp-lambda",
        type=float,
        default=argparse.SUPPRESS,
        help="L2-SP regularization strength: penalizes drift from the --adapt-from "
        "checkpoint's weights (default: 0.0, meaning off)",
    )
    args = parser.parse_args()

    cfg = TrainConfig()
    
    if args.config_file and args.config_file.endswith('.py'):
        print(f"Overriding config using nanoGPT-style file: {args.config_file}")
        with open(args.config_file, "r") as f:
            config_code = f.read()
        
        local_namespace = {}
        exec(config_code, {}, local_namespace)
        
        for key, value in local_namespace.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
                
        if isinstance(cfg.data_dir, str):
            cfg.data_dir = Path(cfg.data_dir)
        if isinstance(cfg.out_dir, str):
            cfg.out_dir = Path(cfg.out_dir)
    
    if hasattr(args, "data_dir"): cfg.data_dir = args.data_dir
    if hasattr(args, "out_dir"): cfg.out_dir = args.out_dir
    if hasattr(args, "device"): cfg.device = args.device
    if hasattr(args, "embedding_dim"): cfg.embedding_dim = args.embedding_dim
    if hasattr(args, "n_heads"): cfg.n_heads = args.n_heads
    if hasattr(args, "n_layers"): cfg.n_layers = args.n_layers
    if hasattr(args, "batch_size"): cfg.batch_size = args.batch_size
    if hasattr(args, "epochs"): cfg.epochs = args.epochs
    if hasattr(args, "lr"): cfg.lr = args.lr
    if hasattr(args, "seed"): cfg.seed = args.seed
    if hasattr(args, "accuracy_interval"): cfg.accuracy_interval = args.accuracy_interval
    if hasattr(args, "adapt_from"): cfg.adapt_from = args.adapt_from
    if hasattr(args, "l2sp_lambda"): cfg.l2sp_lambda = args.l2sp_lambda

    if cfg.accuracy_interval < 1:
        raise ValueError("--accuracy-interval must be >= 1")

    if cfg.l2sp_lambda < 0:
        raise ValueError(
            "--l2sp-lambda must be >= 0 (a negative value has no valid "
            "interpretation -- it would reward drift from the checkpoint "
            "instead of penalizing it)."
        )

    if cfg.l2sp_lambda > 0 and cfg.adapt_from is None:
        raise ValueError(
            "--l2sp-lambda > 0 requires --adapt-from: there is no prior "
            "checkpoint snapshot to penalize drift from otherwise."
        )

    return cfg


def load_adapt_state_dict(checkpoint_path: Path, device: str) -> dict[str, torch.Tensor]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f" Model adaptation checkpoint not found: {checkpoint_path}")

    raw = torch.load(checkpoint_path, map_location=device)
    if isinstance(raw, dict) and "state_dict" in raw:
        state_dict = raw["state_dict"]
    elif isinstance(raw, dict):
        state_dict = raw
    else:
        raise TypeError(f"Unsupported checkpoint format in {checkpoint_path}")

    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError(f"Checkpoint state_dict is empty or invalid: {checkpoint_path}")

    return state_dict


def main() -> None:
    cfg = parse_args()
    
    # Track total runtime
    total_start_time = time.perf_counter()

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)

    # --- Vocabulary ---
    vocab = Vocabulary.from_pickle(cfg.data_dir / "meta.pkl")

    # --- Datasets & loaders ---
    train_dataset = SequenceDataset(cfg.data_dir / "train.bin", vocab)
    val_dataset = SequenceDataset(cfg.data_dir / "val.bin", vocab)

    seq_len = train_dataset.max_len
    print(f"Sequence length : {seq_len} tokens")
    # We use seq_len - 1 because training is next-token prediction:
    # inputs are tokens [0..n-2] and targets are shifted tokens [1..n-1].
    print(f"Batch shape     : ({cfg.batch_size}, {seq_len - 1})")

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True
    )
    train_eval_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.batch_size, shuffle=False
    )

    # --- Model, optimiser, loss ---
    model = CompletionTransformer(
        vocab_size=vocab.vocab_size,
        seq_len=seq_len,
        d_model=cfg.embedding_dim,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
    ).to(cfg.device)

    if cfg.adapt_from is not None:
        state_dict = load_adapt_state_dict(cfg.adapt_from, cfg.device)
        try:
            model.load_state_dict(state_dict)
        except RuntimeError as exc:
            raise ValueError(
                "Model adaptation checkpoint is incompatible with the current model or vocabulary. "
                "Ensure data/vocab and model dimensions match the source checkpoint."
            ) from exc
        print(f"adapting model from checkpoint: {cfg.adapt_from}")

    # parse_args() already enforces that l2sp_lambda > 0 implies adapt_from is set.
    old_params = None
    if cfg.l2sp_lambda > 0:
        old_params = {name: p.detach().clone() for name, p in model.named_parameters()}

    # Calculate exact parameter count
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Model Parameters: {param_count:,}")

    optimizer = optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    print(f"Training on device: {cfg.device}")

    # Metrics collections for charting
    history_epochs = []
    history_train_loss = []
    history_val_loss = []

    # CSV setup
    csv_path = cfg.out_dir / "metrics.csv"
    csv_file = open(csv_path, mode="w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(
        [
            "Epoch",
            "Train Loss",
            "Val Loss",
            "Train Token Acc",
            "Val Token Acc",
            "Train Seq Acc",
            "Val Seq Acc",
            "Training Time (sec)",
            "Param Count",
        ]
    )

    # --- Training loop ---
    for epoch in range(1, cfg.epochs + 1):
        start_time = time.perf_counter()
        
        train_loss = run_epoch(
            model=model, loader=train_loader, vocab_size=vocab.vocab_size,
            criterion=criterion, optimizer=optimizer, grad_clip=cfg.grad_clip,
            device=cfg.device, log_interval=cfg.log_interval, epoch=epoch, total_epochs=cfg.epochs,
            old_params=old_params, l2sp_lambda=cfg.l2sp_lambda,
        )
        
        epoch_time = time.perf_counter() - start_time
        
        val_loss = run_epoch(
            model=model, loader=val_loader, vocab_size=vocab.vocab_size,
            criterion=criterion, optimizer=None, grad_clip=cfg.grad_clip,
            device=cfg.device, log_interval=cfg.log_interval, epoch=epoch, total_epochs=cfg.epochs,
        )

        should_run_accuracy = (epoch % cfg.accuracy_interval == 0) or (epoch == cfg.epochs)
        train_token_acc = None
        val_token_acc = None
        train_seq_acc = None
        val_seq_acc = None
        if should_run_accuracy:
            train_metrics = evaluate_epoch_metrics(
                model=model,
                loader=train_eval_loader,
                vocab_size=vocab.vocab_size,
                criterion=criterion,
                device=cfg.device,
            )
            val_metrics = evaluate_epoch_metrics(
                model=model,
                loader=val_loader,
                vocab_size=vocab.vocab_size,
                criterion=criterion,
                device=cfg.device,
            )
            train_token_acc = train_metrics.token_accuracy
            val_token_acc = val_metrics.token_accuracy
            train_seq_acc = train_metrics.sequence_accuracy
            val_seq_acc = val_metrics.sequence_accuracy

        # Track history data for every epoch to ensure smooth curves
        history_epochs.append(epoch)
        history_train_loss.append(train_loss)
        history_val_loss.append(val_loss)

        if should_run_accuracy:
            print(f"\n{'=' * 60}")
            print(f"Epoch {epoch} summary | Time: {epoch_time:.2f}s")
            print(f"  Train loss : {train_loss:.4f}")
            print(f"  Val loss   : {val_loss:.4f}")
            print(
                "  Train acc  : "
                f"token={train_token_acc * 100:.2f}% "
                f"seq={train_seq_acc * 100:.2f}%"
            )
            print(
                "  Val acc    : "
                f"token={val_token_acc * 100:.2f}% "
                f"seq={val_seq_acc * 100:.2f}%"
            )
        # else:
        #     print(f"  Accuracy   : skipped (every {cfg.accuracy_interval} epochs)")
            print(f"{'=' * 60}\n")

        # Write data row on 10-epoch intervals, on accuracy epochs, and on the last epoch.
        if epoch % 10 == 0 or should_run_accuracy:
            csv_writer.writerow([
                epoch,
                f"{train_loss:.4f}",
                f"{val_loss:.4f}",
                f"{train_token_acc:.4f}" if train_token_acc is not None else "N/A",
                f"{val_token_acc:.4f}" if val_token_acc is not None else "N/A",
                f"{train_seq_acc:.4f}" if train_seq_acc is not None else "N/A",
                f"{val_seq_acc:.4f}" if val_seq_acc is not None else "N/A",
                f"{epoch_time:.2f}",
                param_count,
            ])
            csv_file.flush() # force write to disk safely

    csv_file.close()
    print(f"Metrics table log updated successfully at: {csv_path}")

    # --- Generate Loss Curve Plot ---
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(history_epochs, history_train_loss, label="Train Loss", color="blue", linewidth=2)
        plt.plot(history_epochs, history_val_loss, label="Val Loss", color="orange", linewidth=2)
        plt.title("Loss Over Epochs", fontsize=14, fontweight="bold")
        plt.xlabel("Epochs", fontsize=12)
        plt.ylabel("Cross Entropy Loss", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(fontsize=12)
        
        plot_path = cfg.out_dir / "loss_chart.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Loss plot chart saved successfully at: {plot_path}")
    except ImportError:
        print("Warning: matplotlib not installed. Skipping plot layout creation.")

    # --- Persist weights ---
    weights_path = cfg.out_dir / "model.pth"
    model_cfg = ModelConfig(
        seq_len=seq_len,
        d_model=cfg.embedding_dim,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
    )
    torch.save(make_checkpoint_payload(model, model_cfg), weights_path)
    print(f"Weights saved to {weights_path}")

    # --- Final Statistics ---
    total_runtime = time.perf_counter() - total_start_time
    final_train_loss = history_train_loss[-1]
    final_val_loss = history_val_loss[-1]
    
    print(f"\n{'=' * 60}")
    print(f"TRAINING COMPLETE")
    print(f"  Total Runtime    : {total_runtime:.2f} seconds")
    print(f"  Final Train Loss : {final_train_loss:.4f}")
    print(f"  Final Val Loss   : {final_val_loss:.4f}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()