# References

本プロジェクトで実際に使用した主要なリソースと参考文献。

---

## Models and Algorithms

### 1. DistilBERT: A distilled version of BERT
**Authors**: Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019)
**Publication**: NeurIPS Workshop on Energy Efficient Machine Learning and Cognitive Computing
**Link**: https://arxiv.org/abs/1910.01108
**Usage**: Transformer fine-tuning model として使用

### 2. Long Short-Term Memory
**Authors**: Hochreiter, S., & Schmidhuber, J. (1997)
**Publication**: Neural Computation, 9(8), 1735-1780
**Link**: https://www.bioinf.jku.at/publications/older/2604.pdf
**Usage**: Baseline BiLSTM model の理論的基礎

---

## Dataset

### 3. SMS Spam Collection Dataset
**Authors**: Almeida, T. A., & Hidalgo, J. M. G. (2012)
**Publication**: Proceedings of the 2012 ACM Symposium on Document Engineering
**Links**:
- HuggingFace: https://huggingface.co/datasets/sms_spam
- Original: https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection

**Usage**: 本プロジェクトのメインデータセット（5,574 SMS messages）

---

## Libraries and Frameworks

### 4. PyTorch: An Imperative Style, High-Performance Deep Learning Library
**Authors**: Paszke, A., Gross, S., Massa, F., et al. (2019)
**Publication**: Advances in Neural Information Processing Systems (NeurIPS)
**Links**: https://pytorch.org/ | https://arxiv.org/abs/1912.01703
**Usage**: BiLSTM model の実装フレームワーク

### 5. HuggingFace Transformers: State-of-the-art Natural Language Processing
**Authors**: Wolf, T., Debut, L., Sanh, V., et al. (2020)
**Publication**: EMNLP: System Demonstrations
**Links**: https://huggingface.co/transformers/ | https://arxiv.org/abs/1910.03771
**Usage**: DistilBERT fine-tuning の実装

### 6. HuggingFace Datasets: A Community Library for Natural Language Processing
**Authors**: Lhoest, Q., et al. (2021)
**Publication**: EMNLP: System Demonstrations
**Links**: https://huggingface.co/datasets/ | https://arxiv.org/abs/2109.02846
**Usage**: データローディングとプリプロセッシング

### 7. scikit-learn: Machine Learning in Python
**Authors**: Pedregosa, F., et al. (2011)
**Publication**: Journal of Machine Learning Research, 12, 2825-2830
**Links**: https://scikit-learn.org/ | https://jmlr.org/papers/v12/pedregosa11a.html
**Usage**: 評価メトリクス（accuracy, precision, recall, F1, confusion matrix）

### 8. Matplotlib: Visualization with Python
**Links**: https://matplotlib.org/
**Usage**: 学習曲線と混同行列の可視化

---

## Additional Resources

### Documentation and Tutorials
- **PyTorch Documentation**: https://pytorch.org/docs/
- **HuggingFace Course**: https://huggingface.co/course/
- **scikit-learn User Guide**: https://scikit-learn.org/stable/user_guide.html

---

**Last Updated**: 2026-02-07
