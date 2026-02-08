"""
Evaluation script for both baseline and transformer models.
"""

import argparse
import json
import os

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer

from src.config import SEED, DEVICE, BASELINE_CONFIG, TRANSFORMER_CONFIG
from src.config import BASELINE_RESULTS_DIR, TRANSFORMER_RESULTS_DIR
from src.data import load_and_split_dataset, create_baseline_dataloaders
from src.data import prepare_transformer_datasets
from src.models.baseline_lstm import BiLSTMClassifier


def evaluate_baseline_model(test_loader, model_path, vocab_size, pad_token_id,
                            config=BASELINE_CONFIG, device=DEVICE):
    """
    Evaluate baseline BiLSTM model on test set.

    Args:
        test_loader: Test DataLoader
        model_path (str): Path to saved model checkpoint
        vocab_size (int): Vocabulary size
        pad_token_id (int): Padding token ID
        config (dict): Model configuration
        device (str): Device to evaluate on

    Returns:
        dict: Evaluation metrics and predictions
    """
    # Load model
    model = BiLSTMClassifier(
        vocab_size=vocab_size,
        emb_dim=config["emb_dim"],
        hidden_dim=config["hidden_dim"],
        num_classes=config["num_classes"],
        pad_token_id=pad_token_id,
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Evaluate
    all_labels = []
    all_preds = []
    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            total_loss += loss.item()

            preds = torch.argmax(logits, dim=-1)
            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

    # Compute metrics
    avg_loss = total_loss / len(test_loader)
    acc = accuracy_score(all_labels, all_preds)
    p, r, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary", zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds)

    return {
        "loss": avg_loss,
        "accuracy": acc,
        "precision": p,
        "recall": r,
        "f1": f1,
        "confusion_matrix": cm.tolist(),
        "y_true": all_labels,
        "y_pred": all_preds,
    }


def evaluate_transformer_model(test_tok, model_dir):
    """
    Evaluate transformer model on test set.

    Args:
        test_tok: Tokenized test dataset
        model_dir (str): Directory containing saved model

    Returns:
        dict: Evaluation metrics and predictions
    """
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # Create trainer for evaluation
    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
    )

    # Predict
    pred_out = trainer.predict(test_tok)
    y_true = pred_out.label_ids
    y_pred = np.argmax(pred_out.predictions, axis=1)

    # Compute metrics
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)

    return {
        "accuracy": acc,
        "precision": p,
        "recall": r,
        "f1": f1,
        "confusion_matrix": cm.tolist(),
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
    }


def save_metrics(metrics, save_path):
    """Save metrics to JSON file."""
    # Convert numpy arrays to lists for JSON serialization
    metrics_clean = {}
    for k, v in metrics.items():
        if isinstance(v, np.ndarray):
            metrics_clean[k] = v.tolist()
        elif isinstance(v, (list, float, int, str)):
            metrics_clean[k] = v
        else:
            metrics_clean[k] = str(v)

    with open(save_path, "w") as f:
        json.dump(metrics_clean, f, indent=2)

    print(f"Metrics saved to {save_path}")


def plot_confusion_matrix(cm, save_path, title="Confusion Matrix"):
    """Plot and save confusion matrix."""
    cm_array = np.array(cm)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm_array,
        display_labels=["ham", "spam"]
    )

    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, values_format="d")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"Confusion matrix saved to {save_path}")


def main(args):
    """Main evaluation function."""
    # Set random seed
    torch.manual_seed(SEED)

    # Load dataset
    print("Loading dataset...")
    train_ds, val_ds, test_ds = load_and_split_dataset()

    if args.model_type == "baseline":
        # Evaluate baseline model
        print("\nEvaluating baseline model...")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

        # Create test dataloader
        _, _, test_loader = create_baseline_dataloaders(
            train_ds, val_ds, test_ds, tokenizer,
            batch_size=BASELINE_CONFIG["batch_size"],
            max_length=BASELINE_CONFIG["max_length"],
        )

        # Model path
        model_path = os.path.join(args.model_dir or BASELINE_RESULTS_DIR, "best_model.pt")

        # Evaluate
        metrics = evaluate_baseline_model(
            test_loader,
            model_path,
            vocab_size=tokenizer.vocab_size,
            pad_token_id=tokenizer.pad_token_id,
        )

        # Save results
        save_dir = args.save_dir or BASELINE_RESULTS_DIR
        os.makedirs(save_dir, exist_ok=True)

    elif args.model_type == "transformer":
        # Evaluate transformer model
        print("\nEvaluating transformer model...")

        # Load tokenizer
        model_dir = args.model_dir or TRANSFORMER_RESULTS_DIR

        # Check if model_dir contains a checkpoint directory
        if not os.path.exists(os.path.join(model_dir, "config.json")):
            # Look for checkpoint directories
            checkpoint_dirs = [d for d in os.listdir(model_dir)
                             if d.startswith("checkpoint-") and
                             os.path.isdir(os.path.join(model_dir, d))]
            if checkpoint_dirs:
                # Use the latest checkpoint (highest number)
                checkpoint_dirs.sort(key=lambda x: int(x.split("-")[1]))
                model_dir = os.path.join(model_dir, checkpoint_dirs[-1])
                print(f"Using checkpoint: {model_dir}")

        tokenizer = AutoTokenizer.from_pretrained(model_dir)

        # Prepare test dataset
        _, _, test_tok = prepare_transformer_datasets(
            train_ds, val_ds, test_ds, tokenizer,
            max_length=TRANSFORMER_CONFIG["max_length"],
        )

        # Evaluate
        metrics = evaluate_transformer_model(test_tok, model_dir)

        # Save results
        save_dir = args.save_dir or TRANSFORMER_RESULTS_DIR
        os.makedirs(save_dir, exist_ok=True)

    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    # Print metrics
    print("\nTest Metrics:")
    for k, v in metrics.items():
        if k not in ["confusion_matrix", "y_true", "y_pred"]:
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\nConfusion Matrix:")
    print(np.array(metrics["confusion_matrix"]))

    # Save metrics
    metrics_path = os.path.join(save_dir, "test_metrics.json")
    save_metrics(metrics, metrics_path)

    # Plot confusion matrix
    cm_path = os.path.join(save_dir, "confusion_matrix.png")
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        cm_path,
        title=f"{args.model_type.capitalize()} Confusion Matrix"
    )

    print("\nEvaluation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument(
        "--model-type",
        type=str,
        required=True,
        choices=["baseline", "transformer"],
        help="Model type to evaluate"
    )
    parser.add_argument("--model-dir", type=str, default=None, help="Model directory")
    parser.add_argument("--save-dir", type=str, default=None, help="Save directory")

    args = parser.parse_args()
    main(args)
