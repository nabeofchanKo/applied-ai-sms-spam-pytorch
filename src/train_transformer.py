"""
Training script for transformer model (DistilBERT).
"""

import argparse
import inspect
import json
import os

import numpy as np
import torch
import evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

from src.config import SEED, TRANSFORMER_CONFIG, TRANSFORMER_RESULTS_DIR
from src.data import load_and_split_dataset, prepare_transformer_datasets


def make_training_args(**kwargs):
    """
    Build TrainingArguments in a version-robust way.

    Args:
        **kwargs: Training arguments

    Returns:
        TrainingArguments: Training configuration
    """
    sig = inspect.signature(TrainingArguments.__init__)
    valid = set(sig.parameters.keys())
    filtered = {k: v for k, v in kwargs.items() if k in valid}
    dropped = sorted([k for k in kwargs.keys() if k not in valid])

    if dropped:
        print("Dropped unsupported TrainingArguments keys:", dropped)

    return TrainingArguments(**filtered)


def get_compute_metrics():
    """
    Create metric computation function for Trainer.

    Returns:
        callable: Metric computation function
    """
    accuracy = evaluate.load("accuracy")
    precision = evaluate.load("precision")
    recall = evaluate.load("recall")
    f1 = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)

        return {
            "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
            "precision": precision.compute(
                predictions=preds, references=labels, average="binary"
            )["precision"],
            "recall": recall.compute(
                predictions=preds, references=labels, average="binary"
            )["recall"],
            "f1": f1.compute(
                predictions=preds, references=labels, average="binary"
            )["f1"],
        }

    return compute_metrics


def train_transformer_model(
    train_tok,
    val_tok,
    tokenizer,
    config=TRANSFORMER_CONFIG,
    save_dir=TRANSFORMER_RESULTS_DIR,
):
    """
    Train transformer model using HuggingFace Trainer.

    Args:
        train_tok: Tokenized training dataset
        val_tok: Tokenized validation dataset
        tokenizer: HuggingFace tokenizer
        config (dict): Model configuration
        save_dir (str): Directory to save results

    Returns:
        Trainer: Trained Trainer object
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    # Disable W&B logging
    os.environ["WANDB_DISABLED"] = "true"

    # Load model
    print(f"Loading model: {config['model_name']}")
    model = AutoModelForSequenceClassification.from_pretrained(
        config["model_name"],
        num_labels=config["num_labels"],
    )

    # Prepare training arguments
    args_dict = dict(
        output_dir=save_dir,
        learning_rate=config["learning_rate"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        num_train_epochs=config["num_epochs"],
        weight_decay=config["weight_decay"],
        warmup_ratio=config["warmup_ratio"],
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_strategy="epoch",
        save_total_limit=1,
        seed=SEED,
    )

    # Add evaluation/save strategy with compatibility
    sig_keys = set(inspect.signature(TrainingArguments.__init__).parameters.keys())
    if "evaluation_strategy" in sig_keys:
        args_dict["evaluation_strategy"] = "epoch"
    elif "eval_strategy" in sig_keys:
        args_dict["eval_strategy"] = "epoch"

    if "save_strategy" in sig_keys:
        args_dict["save_strategy"] = "epoch"

    training_args = make_training_args(**args_dict)

    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        processing_class=tokenizer,
        compute_metrics=get_compute_metrics(),
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=config["early_stopping_patience"])
        ],
    )

    # Train
    print("Starting training...")
    train_result = trainer.train()

    # Save training history
    log_history = trainer.state.log_history
    history_path = os.path.join(save_dir, "history.json")
    with open(history_path, "w") as f:
        json.dump(log_history, f, indent=2)
    print(f"Training history saved to {history_path}")

    return trainer


def main(args):
    """Main training function."""
    # Set random seed
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Load and split dataset
    print("Loading dataset...")
    train_ds, val_ds, test_ds = load_and_split_dataset()

    # Load tokenizer
    print("Loading tokenizer...")
    model_name = args.model_name or TRANSFORMER_CONFIG["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Prepare datasets
    print("Tokenizing datasets...")
    train_tok, val_tok, test_tok = prepare_transformer_datasets(
        train_ds,
        val_ds,
        test_ds,
        tokenizer,
        max_length=args.max_length or TRANSFORMER_CONFIG["max_length"],
    )

    # Update config if args provided
    config = TRANSFORMER_CONFIG.copy()
    if args.model_name:
        config["model_name"] = args.model_name
    if args.epochs:
        config["num_epochs"] = args.epochs
    if args.lr:
        config["learning_rate"] = args.lr
    if args.batch_size:
        config["batch_size"] = args.batch_size
    if args.max_length:
        config["max_length"] = args.max_length

    # Train model
    trainer = train_transformer_model(
        train_tok,
        val_tok,
        tokenizer,
        config=config,
        save_dir=args.save_dir or TRANSFORMER_RESULTS_DIR,
    )

    print("\nTraining complete!")
    print(f"Model saved to {args.save_dir or TRANSFORMER_RESULTS_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train transformer model")
    parser.add_argument("--model-name", type=str, default=None, help="Model name")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--max-length", type=int, default=None, help="Max sequence length")
    parser.add_argument("--save-dir", type=str, default=None, help="Save directory")

    args = parser.parse_args()
    main(args)
