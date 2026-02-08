"""
Data loading and preprocessing utilities for SMS spam classification.
"""

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.config import SEED, DATASET_NAME


def load_and_split_dataset(dataset_name=DATASET_NAME, seed=SEED):
    """
    Load SMS spam dataset and split into train/val/test sets.

    Args:
        dataset_name (str): HuggingFace dataset name
        seed (int): Random seed for reproducibility

    Returns:
        tuple: (train_ds, val_ds, test_ds)
    """
    # Load dataset
    ds = load_dataset(dataset_name)

    # 80/20 split (train / temp)
    split1 = ds["train"].train_test_split(
        test_size=0.2,
        seed=seed,
        stratify_by_column="label",
    )
    train_ds = split1["train"]
    temp_ds = split1["test"]

    # 10/10 split (val / test) from temp
    split2 = temp_ds.train_test_split(
        test_size=0.5,
        seed=seed,
        stratify_by_column="label",
    )
    val_ds = split2["train"]
    test_ds = split2["test"]

    print(f"Dataset sizes: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    return train_ds, val_ds, test_ds


def create_baseline_collate_fn(tokenizer, max_length=128):
    """
    Create a collate function for baseline model (BiLSTM) DataLoader.

    Args:
        tokenizer: HuggingFace tokenizer
        max_length (int): Maximum sequence length

    Returns:
        callable: Collate function for DataLoader
    """
    def collate_fn(batch):
        """
        Collate function to process a batch of samples.

        Args:
            batch (list of dict): Each element contains 'sms' and 'label'

        Returns:
            dict: Batch dictionary with input_ids, attention_mask, labels
        """
        texts = [x["sms"] for x in batch]
        labels = torch.tensor([x["label"] for x in batch], dtype=torch.long)

        # Tokenize
        enc = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": labels,
        }

    return collate_fn


def create_baseline_dataloaders(train_ds, val_ds, test_ds, tokenizer,
                                batch_size=64, max_length=128):
    """
    Create DataLoaders for baseline model.

    Args:
        train_ds: Training dataset
        val_ds: Validation dataset
        test_ds: Test dataset
        tokenizer: HuggingFace tokenizer
        batch_size (int): Batch size
        max_length (int): Maximum sequence length

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    collate_fn = create_baseline_collate_fn(tokenizer, max_length)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader, test_loader


def prepare_transformer_datasets(train_ds, val_ds, test_ds, tokenizer, max_length=128):
    """
    Tokenize datasets for transformer model using HuggingFace Trainer.

    Args:
        train_ds: Training dataset
        val_ds: Validation dataset
        test_ds: Test dataset
        tokenizer: HuggingFace tokenizer
        max_length (int): Maximum sequence length

    Returns:
        tuple: (train_tok, val_tok, test_tok)
    """
    def preprocess(batch):
        return tokenizer(
            batch["sms"],
            truncation=True,
            padding=False,
            max_length=max_length,
        )

    def tokenize_and_rename(ds):
        # Tokenize
        tok_ds = ds.map(
            preprocess,
            batched=True,
            remove_columns=["sms"],
        )
        # Rename label -> labels for Trainer
        if "label" in tok_ds.column_names:
            tok_ds = tok_ds.rename_column("label", "labels")
        return tok_ds

    train_tok = tokenize_and_rename(train_ds)
    val_tok = tokenize_and_rename(val_ds)
    test_tok = tokenize_and_rename(test_ds)

    return train_tok, val_tok, test_tok


if __name__ == "__main__":
    # Test data loading
    print("Testing data loading...")
    train_ds, val_ds, test_ds = load_and_split_dataset()

    # Test baseline dataloaders
    print("\nTesting baseline dataloaders...")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    train_loader, val_loader, test_loader = create_baseline_dataloaders(
        train_ds, val_ds, test_ds, tokenizer
    )

    batch = next(iter(train_loader))
    print(f"Batch shapes: input_ids={batch['input_ids'].shape}, "
          f"attention_mask={batch['attention_mask'].shape}, "
          f"labels={batch['labels'].shape}")

    print("\nData loading test passed!")
