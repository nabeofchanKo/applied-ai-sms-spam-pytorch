"""
Visualization utilities for training results.
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_baseline_learning_curves(history_path, save_dir):
    """
    Plot learning curves for baseline model.

    Args:
        history_path (str): Path to history.json
        save_dir (str): Directory to save plots
    """
    # Load history
    with open(history_path, "r") as f:
        history = json.load(f)

    epochs = list(range(1, len(history["train"]) + 1))

    train_loss = [m["loss"] for m in history["train"]]
    val_loss = [m["loss"] for m in history["val"]]
    train_f1 = [m["f1"] for m in history["train"]]
    val_f1 = [m["f1"] for m in history["val"]]

    # Plot loss
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_loss, label="train_loss", marker="o")
    plt.plot(epochs, val_loss, label="val_loss", marker="s")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Baseline Learning Curve (Loss)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    loss_path = os.path.join(save_dir, "learning_curve_loss.png")
    plt.savefig(loss_path, dpi=150)
    plt.close()
    print(f"Loss curve saved to {loss_path}")

    # Plot F1
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_f1, label="train_f1", marker="o")
    plt.plot(epochs, val_f1, label="val_f1", marker="s")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.title("Baseline Learning Curve (F1)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    f1_path = os.path.join(save_dir, "learning_curve_f1.png")
    plt.savefig(f1_path, dpi=150)
    plt.close()
    print(f"F1 curve saved to {f1_path}")


def plot_transformer_learning_curves(history_path, save_dir):
    """
    Plot learning curves for transformer model.

    Args:
        history_path (str): Path to history.json
        save_dir (str): Directory to save plots
    """
    # Load history
    with open(history_path, "r") as f:
        log_history = json.load(f)

    df_logs = pd.DataFrame(log_history).dropna(subset=["epoch"]).copy()

    # Plot F1
    if "eval_f1" in df_logs.columns:
        df_f1 = df_logs.dropna(subset=["eval_f1"])

        plt.figure(figsize=(6, 4))
        plt.plot(df_f1["epoch"], df_f1["eval_f1"], label="val_f1", marker="s")
        plt.xlabel("Epoch")
        plt.ylabel("F1 Score")
        plt.title("Transformer Learning Curve (F1)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        f1_path = os.path.join(save_dir, "learning_curve_f1.png")
        plt.savefig(f1_path, dpi=150)
        plt.close()
        print(f"F1 curve saved to {f1_path}")

    # Plot loss
    has_train_loss = "loss" in df_logs.columns
    has_val_loss = "eval_loss" in df_logs.columns

    if has_train_loss or has_val_loss:
        plt.figure(figsize=(6, 4))

        if has_train_loss:
            df_train = df_logs.dropna(subset=["loss"])
            plt.plot(df_train["epoch"], df_train["loss"], label="train_loss", marker="o")

        if has_val_loss:
            df_val = df_logs.dropna(subset=["eval_loss"])
            plt.plot(df_val["epoch"], df_val["eval_loss"], label="val_loss", marker="s")

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Transformer Learning Curve (Loss)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        loss_path = os.path.join(save_dir, "learning_curve_loss.png")
        plt.savefig(loss_path, dpi=150)
        plt.close()
        print(f"Loss curve saved to {loss_path}")


def main(args):
    """Main visualization function."""
    if args.model_type == "baseline":
        print("Plotting baseline learning curves...")
        plot_baseline_learning_curves(args.history_path, args.save_dir)

    elif args.model_type == "transformer":
        print("Plotting transformer learning curves...")
        plot_transformer_learning_curves(args.history_path, args.save_dir)

    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    print("\nVisualization complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize training results")
    parser.add_argument(
        "--model-type",
        type=str,
        required=True,
        choices=["baseline", "transformer"],
        help="Model type"
    )
    parser.add_argument(
        "--history-path",
        type=str,
        required=True,
        help="Path to history.json"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        required=True,
        help="Directory to save plots"
    )

    args = parser.parse_args()
    main(args)
