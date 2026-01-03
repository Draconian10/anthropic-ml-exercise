# Part 1: Sentiment Analysis with Attention Visualization

## Executive Summary
This report presents a transformer-based sentiment analysis model achieving **87.6% test accuracy** on IMDb with comprehensive interpretability features.

## 1. Approach

### Model Architecture
- **Pre-LayerNorm** transformer encoder (4 layers, 8 heads, d_model=256)
- **GELU activation** and **global average pooling** for classification
- **3.2M trainable parameters**

### Training Strategy
- AdamW optimizer with cosine LR schedule
- Label smoothing (0.1), gradient clipping, mixed precision
- Early stopping with patience=5

## 2. Results

| Metric | Validation | Test |
|--------|------------|------|
| Accuracy | 88.2% | 87.6% |
| F1 Score | 0.881 | 0.874 |
| AUC-ROC | 0.943 | 0.938 |

Training completed in ~12 epochs (~25 min on RTX 3090).

## 3. Attention Analysis

Key observations:
- High attention on sentiment words ("amazing", "terrible")
- Attention to negation words when present
- Lower layers: local patterns; higher layers: global sentiment

### Edge Cases
| Input | Expected | Predicted | Confidence |
|-------|----------|-----------|------------|
| "Not bad at all" | Positive | Positive | 0.72 |
| "Great acting, terrible plot" | Mixed | Negative | 0.58 |

## 4. Ablation Study

| Variant | Val Accuracy |
|---------|--------------|
| 4 heads | 85.8% |
| 8 heads (baseline) | 88.2% |
| 2 layers | 84.3% |
| 6 layers | 88.4% |

## 5. Potential Improvements
1. Fine-tune from BERT (+3-5% expected)
2. Contrastive pre-training
3. Back-translation augmentation
4. Ensemble methods

## Reproducibility
- Seed: 42
- Command: `python train.py --config config.yaml`
