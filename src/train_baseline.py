"""
Training script for baseline BiLSTM model.
"""

import argparse
import copy
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoTokenizer

from src.config import SEED, DEVICE, BASELINE_CONFIG, BASELINE_RESULTS_DIR
from src.data import load_and_split_dataset, create_baseline_dataloaders
from src.models.baseline_lstm import BiLSTMClassifier


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    """
    Run one epoch for training or evaluation.

    Args:
        model: PyTorch model
        loader: DataLoader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to run on
        train (bool): Training mode if True, eval mode if False

    Returns:
        dict: Metrics dictionary
    """
    model.train() if train else model.eval()

    all_labels = []
    all_preds = []
    total_loss = 0.0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        if train:
            optimizer.zero_grad()

        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)

        if train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

        preds = torch.argmax(logits, dim=-1)
        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())

    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    p, r, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary", zero_division=0
    )

    return {
        "loss": avg_loss,
        "accuracy": acc,
        "precision": p,
        "recall": r,
        "f1": f1,
    }


def train_baseline_model(
    train_loader,
    val_loader,
    vocab_size,
    pad_token_id,
    config=BASELINE_CONFIG,
    device=DEVICE,
    save_dir=BASELINE_RESULTS_DIR,
):
    """
    Train baseline BiLSTM model.

    Args:
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        vocab_size (int): Vocabulary size
        pad_token_id (int): Padding token ID
        config (dict): Model configuration
        device (str): Device to train on
        save_dir (str): Directory to save results

    Returns:
        tuple: (model, history)
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    # Initialize model
    model = BiLSTMClassifier(
        vocab_size=vocab_size,
        emb_dim=config["emb_dim"],
        hidden_dim=config["hidden_dim"],
        num_classes=config["num_classes"],
        pad_token_id=pad_token_id,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"])

    # Training loop
    epochs = config["num_epochs"]
    history = {"train": [], "val": []}
    best_val_f1 = -1.0
    best_state = None
    best_epoch = None

    print(f"Training on {device}...")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_metrics = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        print(
            f"[Epoch {epoch}/{epochs}] "
            f"train_loss={train_metrics['loss']:.4f} train_f1={train_metrics['f1']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} val_f1={val_metrics['f1']:.4f} "
            f"(val_P={val_metrics['precision']:.4f}, val_R={val_metrics['recall']:.4f}, "
            f"val_acc={val_metrics['accuracy']:.4f})"
        )

        # Save best checkpoint
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch

    print(f"\nBest checkpoint: epoch={best_epoch}, val_f1={best_val_f1:.4f}")

    # Load best checkpoint
    model.load_state_dict(best_state)

    # Save model
    model_path = os.path.join(save_dir, "best_model.pt")
    torch.save(best_state, model_path)
    print(f"Model saved to {model_path}")

    # Save training history
    history_path = os.path.join(save_dir, "history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to {history_path}")

    return model, history


def main(args):
    """Main training function."""
    # Set random seed
    torch.manual_seed(SEED)

    # Load and split dataset
    print("Loading dataset...")
    train_ds, val_ds, test_ds = load_and_split_dataset()

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name or BASELINE_CONFIG.get("tokenizer_name", "distilbert-base-uncased")
    )

    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_baseline_dataloaders(
        train_ds,
        val_ds,
        test_ds,
        tokenizer,
        batch_size=args.batch_size or BASELINE_CONFIG["batch_size"],
        max_length=args.max_length or BASELINE_CONFIG["max_length"],
    )

    # Update config if args provided
    config = BASELINE_CONFIG.copy()
    if args.epochs:
        config["num_epochs"] = args.epochs
    if args.lr:
        config["learning_rate"] = args.lr

    # Train model
    model, history = train_baseline_model(
        train_loader,
        val_loader,
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_token_id,
        config=config,
        device=DEVICE,
        save_dir=args.save_dir or BASELINE_RESULTS_DIR,
    )

    print("\nTraining complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train baseline BiLSTM model")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--max-length", type=int, default=None, help="Max sequence length")
    parser.add_argument("--tokenizer-name", type=str, default=None, help="Tokenizer name")
    parser.add_argument("--save-dir", type=str, default=None, help="Save directory")

    args = parser.parse_args()
    main(args)
