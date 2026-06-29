# SMS Spam Classification with Baseline NN and Transformer (PyTorch)

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Overview

This project implements and compares two neural approaches for SMS spam classification:
1. **Baseline BiLSTM**: A custom neural network built from scratch using PyTorch
2. **DistilBERT**: A transformer-based model fine-tuned with HuggingFace Transformers

The goal is to demonstrate how model architecture choices affect performance, error patterns, and robustness in short-text classification tasks.

**Key Results:**
- BiLSTM: **98.4% accuracy**, **94.0% F1 score**
- DistilBERT: **98.6% accuracy**, **94.5% F1 score**

---

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Results](#results)
- [Documentation](#documentation)
- [Testing](#testing)
- [License](#license)

---

## Features

✅ **Two Model Architectures**: BiLSTM baseline and DistilBERT transformer
✅ **Modular Code**: Clean separation of data, models, training, and evaluation
✅ **CLI Support**: Train and evaluate models from command line
✅ **Comprehensive Tests**: Unit tests for data loading and models
✅ **Detailed Documentation**: Technical report and references
✅ **Reproducible**: Fixed random seeds and version-controlled dependencies

---

## Project Structure

```
applied-ai-sms-spam-pytorch/
├── src/                          # Source code
│   ├── config.py                 # Configuration and hyperparameters
│   ├── data.py                   # Data loading and preprocessing
│   ├── models/
│   │   └── baseline_lstm.py      # BiLSTM model architecture
│   ├── train_baseline.py         # Baseline training script
│   ├── train_transformer.py      # Transformer training script
│   ├── eval.py                   # Evaluation script
│   └── visualize.py              # Result visualization
├── notebooks/                    # Jupyter notebooks
│   ├── 01_setup_and_data.ipynb  # Data exploration and baseline
│   └── 02_transformer_finetuning.ipynb  # Transformer fine-tuning
├── tests/                        # Unit tests
│   ├── test_data.py              # Data loading tests
│   └── test_models.py            # Model architecture tests
├── docs/                         # Documentation
│   ├── report.md                 # Technical report
│   └── references.md             # References and citations
├── results/                      # Experimental results
│   ├── baseline/                 # Baseline model results
│   ├── transformer/              # Transformer model results
│   └── error_examples.md         # Error analysis
├── data/                         # Dataset (auto-downloaded)
├── requirements.txt              # Python dependencies
├── LICENSE                       # MIT License
└── README.md                     # This file
```

---

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (optional, for faster training)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/nabeofchanko/applied-ai-sms-spam-pytorch.git
cd applied-ai-sms-spam-pytorch
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Quick Start

### Option 1: Jupyter Notebooks (Recommended for exploration)

Run the notebooks in order:
```bash
jupyter notebook notebooks/01_setup_and_data.ipynb
jupyter notebook notebooks/02_transformer_finetuning.ipynb
```

### Option 2: Command Line (Recommended for production)

**Train baseline model:**
```bash
python -m src.train_baseline
```

**Train transformer model:**
```bash
python -m src.train_transformer
```

**Evaluate models:**
```bash
python -m src.eval --model-type baseline
python -m src.eval --model-type transformer
```

---

## Usage

### Training

#### Baseline BiLSTM

```bash
# Train with default settings
python -m src.train_baseline

# Train with custom hyperparameters
python -m src.train_baseline --epochs 10 --lr 0.001 --batch-size 32
```

#### DistilBERT Transformer

```bash
# Train with default settings
python -m src.train_transformer

# Train with custom hyperparameters
python -m src.train_transformer --epochs 5 --lr 2e-5 --batch-size 16
```

### Evaluation

```bash
# Evaluate baseline model
python -m src.eval --model-type baseline --model-dir results/baseline

# Evaluate transformer model
python -m src.eval --model-type transformer --model-dir results/transformer
```

### Visualization

```bash
# Visualize baseline learning curves
python -m src.visualize --model-type baseline \
  --history-path results/baseline/history.json \
  --save-dir results/baseline

# Visualize transformer learning curves
python -m src.visualize --model-type transformer \
  --history-path results/transformer/history.json \
  --save-dir results/transformer
```

---

## Results

### Quantitative Comparison

| Metric | Baseline BiLSTM | DistilBERT | Improvement |
|--------|-----------------|------------|-------------|
| **Accuracy** | 98.39% | 98.57% | +0.18% |
| **Precision** | 94.59% | 97.18% | +2.59% |
| **Recall** | 93.33% | 92.00% | -1.33% |
| **F1 Score** | 93.96% | 94.52% | +0.56% |

### Confusion Matrices

**Baseline BiLSTM:**
```
              Predicted
              Ham   Spam
Actual  Ham   479    4
        Spam    5   70
```

**DistilBERT:**
```
              Predicted
              Ham   Spam
Actual  Ham   481    2
        Spam    6   69
```

### Key Insights

- ✅ **Both models achieve excellent performance** (>98% accuracy)
- ✅ **Transformer trades recall for precision**: false positives drop (4 → 2) while false negatives rise slightly (5 → 6)
- ✅ **BiLSTM is lightweight and fast** (~1.2M parameters vs. 66M)
- ✅ **Trade-off**: Accuracy vs. model size/inference speed

For detailed analysis, see [docs/report.md](docs/report.md).

---

## Documentation

- **[Technical Report](docs/report.md)**: Comprehensive analysis of experiments, results, and insights
- **[References](docs/references.md)**: Academic papers, datasets, and resources
- **[Error Analysis](results/error_examples.md)**: Examples of model errors and patterns

---

## Testing

Run tests to verify the implementation:

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_data.py -v
pytest tests/test_models.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

---

## Dataset

This project uses the **SMS Spam Collection** dataset:
- **Source**: [HuggingFace Datasets](https://huggingface.co/datasets/sms_spam)
- **Size**: 5,574 SMS messages
- **Classes**: Ham (legitimate) and Spam
- **License**: Public domain

The dataset is automatically downloaded when running the code.

---

## Reproducibility

All experiments use fixed random seeds (SEED=42) for reproducibility:
- PyTorch: `torch.manual_seed(42)`
- NumPy: `np.random.seed(42)`
- Data splits: Stratified with `random_state=42`

Hardware used for experiments:
- GPU: Tesla T4 (Google Colab)
- PyTorch: 2.9.0+cu126

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{sms-spam-classification,
  title={SMS Spam Classification with BiLSTM and Transformer},
  author={Applied AI SMS Spam Classification Project},
  year={2026},
  url={https://github.com/nabeofchanko/applied-ai-sms-spam-pytorch}
}
```

---

## Acknowledgments

- **SMS Spam Collection**: Almeida, T.A. & Hidalgo, J.M.G.
- **HuggingFace**: For transformers and datasets libraries
- **PyTorch**: For deep learning framework

---

## Contact

For questions or feedback, please open an issue on GitHub.
