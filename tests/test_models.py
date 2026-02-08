"""
Tests for model architectures.
"""

import pytest
import torch

from src.models.baseline_lstm import BiLSTMClassifier


class TestBiLSTMClassifier:
    """Test BiLSTM classifier model."""

    @pytest.fixture
    def model(self):
        """Create a test model instance."""
        return BiLSTMClassifier(
            vocab_size=1000,
            emb_dim=64,
            hidden_dim=64,
            num_classes=2,
            pad_token_id=0,
        )

    def test_model_initialization(self, model):
        """Test that model initializes correctly."""
        assert isinstance(model, BiLSTMClassifier)
        assert model.embedding.num_embeddings == 1000
        assert model.embedding.embedding_dim == 64
        assert model.lstm.hidden_size == 64
        assert model.classifier.out_features == 2

    def test_forward_pass_shape(self, model):
        """Test that forward pass produces correct output shape."""
        batch_size = 4
        seq_length = 20

        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        attention_mask = torch.ones(batch_size, seq_length)

        logits = model(input_ids, attention_mask)

        assert logits.shape == (batch_size, 2)

    def test_forward_pass_dtypes(self, model):
        """Test that forward pass produces correct dtype."""
        batch_size = 4
        seq_length = 20

        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        attention_mask = torch.ones(batch_size, seq_length)

        logits = model(input_ids, attention_mask)

        assert logits.dtype == torch.float32

    def test_model_with_padding(self, model):
        """Test that model handles padding correctly."""
        batch_size = 4
        seq_length = 20

        # Create inputs with padding
        input_ids = torch.randint(1, 1000, (batch_size, seq_length))
        input_ids[:, 10:] = 0  # Pad second half

        attention_mask = torch.ones(batch_size, seq_length)
        attention_mask[:, 10:] = 0  # Mask padding

        # Forward pass should work without errors
        logits = model(input_ids, attention_mask)

        assert logits.shape == (batch_size, 2)
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()

    def test_model_gradient_flow(self, model):
        """Test that gradients flow through the model."""
        batch_size = 4
        seq_length = 20

        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        attention_mask = torch.ones(batch_size, seq_length)
        labels = torch.randint(0, 2, (batch_size,))

        # Forward pass
        logits = model(input_ids, attention_mask)

        # Compute loss
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(logits, labels)

        # Backward pass
        loss.backward()

        # Check that gradients exist
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_model_parameter_count(self, model):
        """Test that model has reasonable number of parameters."""
        total_params = sum(p.numel() for p in model.parameters())

        # With vocab_size=1000, emb_dim=64, hidden_dim=64:
        # - Embedding: 1000 * 64 = 64,000
        # - LSTM: ~4 * (64 * 64 + 64 * 128 + 128) ≈ 49,664
        # - Classifier: 128 * 2 + 2 = 258
        # Total: ~114,000

        assert 100_000 < total_params < 150_000

    def test_model_eval_mode(self, model):
        """Test that model can be set to eval mode."""
        model.eval()

        batch_size = 4
        seq_length = 20

        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        attention_mask = torch.ones(batch_size, seq_length)

        with torch.no_grad():
            logits = model(input_ids, attention_mask)

        assert logits.shape == (batch_size, 2)


class TestModelDeviceCompatibility:
    """Test model compatibility with different devices."""

    def test_model_on_cpu(self):
        """Test that model works on CPU."""
        model = BiLSTMClassifier(
            vocab_size=100,
            emb_dim=32,
            hidden_dim=32,
            num_classes=2,
        )

        model = model.to("cpu")

        input_ids = torch.randint(0, 100, (2, 10)).to("cpu")
        attention_mask = torch.ones(2, 10).to("cpu")

        logits = model(input_ids, attention_mask)

        assert logits.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_model_on_cuda(self):
        """Test that model works on CUDA."""
        model = BiLSTMClassifier(
            vocab_size=100,
            emb_dim=32,
            hidden_dim=32,
            num_classes=2,
        )

        model = model.to("cuda")

        input_ids = torch.randint(0, 100, (2, 10)).to("cuda")
        attention_mask = torch.ones(2, 10).to("cuda")

        logits = model(input_ids, attention_mask)

        assert logits.device.type == "cuda"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
