"""
Configuration file for SMS spam classification project.
"""

import torch

# ============================================================================
# General Settings
# ============================================================================
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================================
# Dataset Settings
# ============================================================================
DATASET_NAME = "sms_spam"
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# ============================================================================
# Baseline Model (BiLSTM) Settings
# ============================================================================
BASELINE_CONFIG = {
    "emb_dim": 128,
    "hidden_dim": 128,
    "num_classes": 2,
    "max_length": 128,
    "batch_size": 64,
    "learning_rate": 2e-3,
    "num_epochs": 8,
    "patience": 2,  # Early stopping patience
}

# ============================================================================
# Transformer Model Settings
# ============================================================================
TRANSFORMER_CONFIG = {
    "model_name": "distilbert-base-uncased",
    "max_length": 128,
    "num_labels": 2,
    "batch_size": 16,
    "learning_rate": 2e-5,
    "num_epochs": 3,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "early_stopping_patience": 1,
}

# ============================================================================
# Paths
# ============================================================================
RESULTS_DIR = "results"
BASELINE_RESULTS_DIR = f"{RESULTS_DIR}/baseline"
TRANSFORMER_RESULTS_DIR = f"{RESULTS_DIR}/transformer"
