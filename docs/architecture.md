# Architecture Documentation

## System Overview

TruthLens AI implements a cross-modal consistency-aware multimodal fake news detection system that analyzes both text and images using deep learning.

## Model Architecture

### Encoder Ensemble

The system uses three pretrained encoders:

1. **BERT (Text Encoder)**
   - Model: `google-bert/bert-base-uncased`
   - Input: Tokenized text (max 128 tokens)
   - Output: 768-dimensional CLS embedding
   - Role: Extract semantic text features

2. **Vision Transformer (ViT) (Image Encoder)**
   - Model: `google/vit-base-patch16-224`
   - Input: 224x224 RGB images
   - Output: 768-dimensional CLS embedding
   - Role: Extract visual features from images

3. **CLIP (Semantic Similarity)**
   - Model: `openai/clip-vit-base-patch32`
   - Input: Text and images
   - Output: 512-dimensional normalized embeddings
   - Role: Measure semantic alignment between text and images

### Feature Projection

Both BERT and ViT produce 768-dimensional embeddings. These are projected to a 256-dimensional space using trainable projection layers:

```
T_projected = Linear(768, 256)(T)
I_projected = Linear(768, 256)(I)
```

### Fusion Strategy

The fusion module combines multiple interaction signals:

1. **Projected text features** (256 dimensions)
2. **Projected image features** (256 dimensions)
3. **Absolute difference** |T_projected - I_projected| (256 dimensions)
4. **Element-wise product** T_projected ⊙ I_projected (256 dimensions)
5. **CLIP cosine similarity** (1 dimension)

Total fusion vector: 256 + 256 + 256 + 256 + 1 = 1025 dimensions

### Classification Head

```
Fusion Vector (1025) → Linear(1025, 256) → ReLU → Dropout(0.3)
                   → Linear(256, 128) → ReLU → Dropout(0.3)
                   → Linear(128, 1)
```

Output: Single logit (sigmoid applied during inference)

## Model Variants

### Model A: Text-Only BERT
- Uses only BERT text features
- Classification head directly on BERT CLS embedding
- Baseline for text-only performance

### Model B: Image-Only ViT
- Uses only ViT image features
- Classification head directly on ViT CLS embedding
- Baseline for image-only performance

### Model C: Standard Multimodal Fusion
- Concatenates projected text and image features
- No explicit consistency features
- Baseline for multimodal without consistency

### Model D: Consistency-Aware Fusion (Proposed)
- Full fusion with difference, product, and CLIP similarity
- Tests hypothesis that consistency improves classification

### Model E: Ablation (No CLIP Similarity)
- Same as Model D but without CLIP similarity
- Tests contribution of explicit consistency signal

## Data Flow

```
Input Text → BERT Tokenizer → BERT Encoder → 768-dim CLS
Input Image → ViT Preprocessor → ViT Encoder → 768-dim CLS

768-dim Text → Projection → 256-dim Text
768-dim Image → Projection → 256-dim Image

Text + Image → CLIP Encoder → CLIP Similarity Score

[256-dim Text, 256-dim Image, 256-dim Diff, 256-dim Product, 1-dim Sim]
           → Concatenation → 1025-dim Fusion Vector
           → Classification Head → Logit → Prediction
```

## Training Procedure

1. **Frozen Encoders**: Pretrained encoders are frozen during initial training
2. **Only Fusion Head Trained**: Only projection layers and classification head have trainable weights
3. **BCEWithLogitsLoss**: Binary cross-entropy loss for fake/real classification
4. **AdamW Optimizer**: Learning rate 1e-3, weight decay 1e-4
5. **Early Stopping**: Based on validation Macro-F1

## Feature Caching

To speed up training with frozen encoders:

1. Extract features for all samples once
2. Cache text features, image features, and CLIP similarities
3. Train fusion head on cached features
4. Invalidate cache only when model/preprocessing changes

## Tensor Dimensions Summary

| Component | Input Dim | Output Dim |
|-----------|-----------|------------|
| BERT | [B, 128] tokens | [B, 768] |
| ViT | [B, 3, 224, 224] | [B, 768] |
| CLIP | Text/Image | [B, 512] (normalized) |
| Text Projection | [B, 768] | [B, 256] |
| Image Projection | [B, 768] | [B, 256] |
| Fusion | [B, 1025] | [B, 1] |
