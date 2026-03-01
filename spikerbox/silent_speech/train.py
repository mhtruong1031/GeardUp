"""
Train the Silent Speech 1D CNN on collected .npz data.

Usage:
  python -m spikerbox.silent_speech.train --data data/silent_speech --out models/silent_speech/silent_speech_cnn.pt
"""

import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import DEFAULT_DATA_DIR, DEFAULT_MODEL_PATH
from .dataset import build_train_val_splits
from .model import SilentSpeechCNN


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Silent Speech 1D CNN")
    parser.add_argument("--data", default=DEFAULT_DATA_DIR, help="Directory of .npz collected data")
    parser.add_argument("--out", default=DEFAULT_MODEL_PATH, help="Output path for saved model")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-cuda", action="store_true", help="Disable CUDA (use CPU)")
    parser.add_argument("--no-bandpass", action="store_true", help="Skip bandpass filter in dataset")
    parser.add_argument("--no-normalize", action="store_true", help="Skip per-channel normalization")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Device: {device}")

    train_ds, val_ds = build_train_val_splits(
        args.data,
        val_ratio=args.val_ratio,
        stratify=True,
        seed=args.seed,
        apply_bandpass=not args.no_bandpass,
        apply_normalize=not args.no_normalize,
    )
    if len(train_ds) == 0:
        raise SystemExit(f"No training samples in {args.data}")
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = SilentSpeechCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)  # (B, num_classes, T')
            logits = out.max(dim=2).values  # MIL: max over time -> (B, num_classes)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pred = logits.argmax(dim=1)
            train_correct += (pred == y).sum().item()
            train_total += y.size(0)

        train_acc = train_correct / train_total if train_total else 0
        train_loss /= len(train_loader) if train_loader else 1

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                logits = out.max(dim=2).values
                pred = logits.argmax(dim=1)
                val_correct += (pred == y).sum().item()
                val_total += y.size(0)

        val_acc = val_correct / val_total if val_total else 0

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
            }
            torch.save(state, args.out)
            print(f"Epoch {epoch + 1}/{args.epochs}  loss={train_loss:.4f}  train_acc={train_acc:.4f}  val_acc={val_acc:.4f}  [saved]")
        else:
            print(f"Epoch {epoch + 1}/{args.epochs}  loss={train_loss:.4f}  train_acc={train_acc:.4f}  val_acc={val_acc:.4f}")

    print(f"Done. Best val accuracy: {best_val_acc:.4f}. Model saved to {args.out}")


if __name__ == "__main__":
    main()
