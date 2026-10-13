# Methodology Documentation

## Research Question

> Does explicitly incorporating image-text semantic consistency improve multimodal fake-news classification compared with standard text-image feature fusion, and how do both approaches behave under controlled image-text mismatch?

## Approach

### Multimodal Feature Extraction

We use three pretrained models to extract features:

1. **BERT**: Extracts contextual text representations
2. **Vision Transformer (ViT)**: Extracts visual representations from images
3. **CLIP**: Measures semantic alignment between text and images

### Consistency-Aware Fusion

Unlike standard multimodal fusion that concatenates features, our approach explicitly models consistency:

1. **Absolute Difference**: |T_proj - I_proj| captures where features disagree
2. **Element-wise Product**: T_proj ⊙ I_proj captures feature agreement
3. **CLIP Similarity**: Direct semantic alignment score

### Hypothesis

We hypothesize that:
- Consistent image-text pairs (high CLIP similarity) are more likely to be legitimate
- Inconsistent pairs (low CLIP similarity) may indicate manipulation or misuse
- The consistency signal provides complementary information to content features

## Experiments

### Baseline Models

| Model | Description | Purpose |
|-------|-------------|---------|
| A | Text-only BERT | Text-only baseline |
| B | Image-only ViT | Image-only baseline |
| C | Standard fusion | Multimodal without consistency |
| D | Consistency-aware | Proposed method |
| E | Ablation | Model D without CLIP similarity |

### Evaluation Metrics

- **Accuracy**: Overall correct predictions
- **Macro-F1**: Balanced F1 across classes
- **Weighted-F1**: Class-weighted F1
- **Precision/Recall**: Per-class metrics
- **ROC-AUC**: Ranking capability

### Image-Swap Experiment

Controlled experiment to test robustness:

1. Keep original text and labels
2. Replace images with random unrelated images
3. Measure prediction changes
4. Compare consistency-aware vs standard models

**Hypothesis**: If consistency-aware model is more robust to swaps, it may be relying less on spurious image-text correlations.

## Ethical Considerations

1. **Semantic ≠ Factual**: CLIP measures semantic alignment, not truthfulness
2. **Dataset Bias**: Distant supervision labels contain noise
3. **Domain Shift**: Reddit data may not generalize to formal news
4. **Misuse Potential**: Any detection system can be gamed

## Limitations

1. **No External Verification**: Model doesn't verify against external sources
2. **Limited Context**: Short text snippets vs full articles
3. **Image Quality**: Corrupt images affect performance
4. **Text Length**: 128 token limit for BERT
