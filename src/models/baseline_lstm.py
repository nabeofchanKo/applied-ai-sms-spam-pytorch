"""
Baseline BiLSTM model for SMS spam classification.
"""

import torch
import torch.nn as nn


class BiLSTMClassifier(nn.Module):
    """
    Bidirectional LSTM classifier for binary text classification.

    Architecture:
        - Embedding layer
        - Bidirectional LSTM encoder
        - Mean pooling over sequence
        - Linear classifier
    """

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 128,
        hidden_dim: int = 128,
        num_classes: int = 2,
        pad_token_id: int = 0,
    ):
        """
        Initialize BiLSTM classifier.

        Args:
            vocab_size (int): Vocabulary size
            emb_dim (int): Embedding dimension
            hidden_dim (int): LSTM hidden dimension
            num_classes (int): Number of output classes
            pad_token_id (int): Padding token ID
        """
        super().__init__()

        # Embedding layer: token ids -> dense vectors
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=emb_dim,
            padding_idx=pad_token_id,  # PAD tokens produce zero vectors
        )

        # Bidirectional LSTM encoder
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )

        # Linear classifier on top of pooled sequence representation
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids, attention_mask):
        """
        Forward pass.

        Args:
            input_ids (Tensor): shape (B, T) - Token IDs
            attention_mask (Tensor): shape (B, T) - Attention mask

        Returns:
            logits (Tensor): shape (B, num_classes) - Classification logits
        """
        # (B, T) -> (B, T, E)
        x = self.embedding(input_ids)

        # (B, T, E) -> (B, T, 2H)
        outputs, _ = self.lstm(x)

        # Masked mean pooling
        # Expand attention mask to (B, T, 1)
        mask = attention_mask.unsqueeze(-1)

        # Zero-out padded positions
        outputs = outputs * mask

        # Sum over time dimension
        summed = outputs.sum(dim=1)

        # Count valid (non-padding) tokens
        lengths = mask.sum(dim=1).clamp(min=1e-6)

        # Compute mean pooled representation
        pooled = summed / lengths

        # (B, 2H) -> (B, num_classes)
        logits = self.classifier(pooled)

        return logits


if __name__ == "__main__":
    # Test model
    print("Testing BiLSTMClassifier...")

    vocab_size = 30522  # DistilBERT vocab size
    batch_size = 8
    seq_length = 128

    model = BiLSTMClassifier(
        vocab_size=vocab_size,
        emb_dim=128,
        hidden_dim=128,
        num_classes=2,
        pad_token_id=0,
    )

    # Create dummy inputs
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length)

    # Forward pass
    logits = model(input_ids, attention_mask)

    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("\nModel test passed!")
