"""
Tests for data loading and preprocessing.
"""

import pytest
import torch
from transformers import AutoTokenizer

from src.data import (
    load_and_split_dataset,
    create_baseline_dataloaders,
    prepare_transformer_datasets,
)


class TestDataLoading:
    """Test data loading functionality."""

    @pytest.fixture(scope="class")
    def datasets(self):
        """Load datasets once for all tests."""
        return load_and_split_dataset()

    def test_dataset_split_sizes(self, datasets):
        """Test that dataset splits have correct sizes."""
        train_ds, val_ds, test_ds = datasets

        total = len(train_ds) + len(val_ds) + len(test_ds)

        # Check approximate split ratios (80/10/10)
        assert 0.75 < len(train_ds) / total < 0.85
        assert 0.08 < len(val_ds) / total < 0.12
        assert 0.08 < len(test_ds) / total < 0.12

    def test_dataset_has_required_columns(self, datasets):
        """Test that datasets have 'sms' and 'label' columns."""
        train_ds, val_ds, test_ds = datasets

        for ds in [train_ds, val_ds, test_ds]:
            assert "sms" in ds.column_names
            assert "label" in ds.column_names

    def test_label_distribution(self, datasets):
        """Test that class distribution is preserved (stratified split)."""
        train_ds, val_ds, test_ds = datasets

        def get_spam_ratio(ds):
            labels = ds["label"]
            return labels.count(1) / len(labels)

        train_ratio = get_spam_ratio(train_ds)
        val_ratio = get_spam_ratio(val_ds)
        test_ratio = get_spam_ratio(test_ds)

        # All splits should have similar spam ratios (within 5%)
        assert abs(train_ratio - val_ratio) < 0.05
        assert abs(train_ratio - test_ratio) < 0.05


class TestBaselineDataLoaders:
    """Test baseline model dataloaders."""

    @pytest.fixture(scope="class")
    def tokenizer(self):
        """Load tokenizer once for all tests."""
        return AutoTokenizer.from_pretrained("distilbert-base-uncased")

    @pytest.fixture(scope="class")
    def dataloaders(self, tokenizer):
        """Create dataloaders once for all tests."""
        train_ds, val_ds, test_ds = load_and_split_dataset()
        return create_baseline_dataloaders(
            train_ds, val_ds, test_ds, tokenizer, batch_size=8, max_length=128
        )

    def test_dataloader_batch_shape(self, dataloaders):
        """Test that dataloader produces correct batch shapes."""
        train_loader, val_loader, test_loader = dataloaders

        batch = next(iter(train_loader))

        assert "input_ids" in batch
        assert "attention_mask" in batch
        assert "labels" in batch

        # Check shapes
        batch_size = batch["input_ids"].shape[0]
        seq_length = batch["input_ids"].shape[1]

        assert batch_size <= 8  # Last batch might be smaller
        assert seq_length == 128
        assert batch["attention_mask"].shape == (batch_size, seq_length)
        assert batch["labels"].shape == (batch_size,)

    def test_dataloader_dtypes(self, dataloaders):
        """Test that batch tensors have correct dtypes."""
        train_loader, _, _ = dataloaders

        batch = next(iter(train_loader))

        assert batch["input_ids"].dtype == torch.long
        assert batch["attention_mask"].dtype == torch.long
        assert batch["labels"].dtype == torch.long


class TestTransformerDatasets:
    """Test transformer model datasets."""

    @pytest.fixture(scope="class")
    def tokenizer(self):
        """Load tokenizer once for all tests."""
        return AutoTokenizer.from_pretrained("distilbert-base-uncased")

    @pytest.fixture(scope="class")
    def tokenized_datasets(self, tokenizer):
        """Create tokenized datasets once for all tests."""
        train_ds, val_ds, test_ds = load_and_split_dataset()
        return prepare_transformer_datasets(
            train_ds, val_ds, test_ds, tokenizer, max_length=128
        )

    def test_tokenized_dataset_columns(self, tokenized_datasets):
        """Test that tokenized datasets have required columns."""
        train_tok, val_tok, test_tok = tokenized_datasets

        for ds in [train_tok, val_tok, test_tok]:
            assert "input_ids" in ds.column_names
            assert "attention_mask" in ds.column_names
            assert "labels" in ds.column_names
            assert "sms" not in ds.column_names  # Raw text should be removed

    def test_tokenized_dataset_lengths(self, tokenized_datasets):
        """Test that tokenization doesn't change dataset sizes."""
        train_ds, val_ds, test_ds = load_and_split_dataset()
        train_tok, val_tok, test_tok = tokenized_datasets

        assert len(train_tok) == len(train_ds)
        assert len(val_tok) == len(val_ds)
        assert len(test_tok) == len(test_ds)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
