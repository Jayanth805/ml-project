# Cross-Modal Consistency-Aware Multimodal Fake News Detection Using Deep Learning

## Project Overview

**TruthLens AI** is a research-oriented system for detecting fake news by analyzing both text and images using deep learning. The system implements a novel consistency-aware multimodal fusion approach that explicitly measures image-text semantic similarity using CLIP and incorporates this signal into a classification decision.

## Research Question

> Does explicitly incorporating image-text semantic consistency improve multimodal fake-news classification compared with standard text-image feature fusion, and how do both approaches behave under controlled image-text mismatch?

## Key Features

- **Text Analysis**: BERT-based text encoding for semantic understanding
- **Image Analysis**: Vision Transformer (ViT) for visual feature extraction
- **Consistency Detection**: CLIP-based image-text semantic similarity measurement
- **Multimodal Fusion**: Advanced fusion combining text features, image features, and consistency signals
- **Image-Swap Experiments**: Controlled evaluation of model robustness to image-text mismatches
- **Ablation Studies**: Systematic ablation to measure the contribution of each component

## Architecture

```
Input Text + Image
       ↓
┌──────┴──────┐
│   BERT      │   ViT         CLIP
│ (768-dim)   │  (768-dim)    (similarity)
│      ↓      │      ↓              ↓
│  Projection │  Projection    Raw Score
│  (256-dim)  │  (256-dim)
│      ↓      │      ↓
└──────┬──────┘
       ↓
  Concatenation + Difference + Product + Similarity
       ↓
  Fusion Classifier (MLP)
       ↓
   REAL / FAKE Prediction
```

## Requirements

- Python 3.9+
- PyTorch 2.0+
- 8GB+ RAM (16GB recommended)
- GPU with 8GB+ VRAM for training (CPU inference supported)
- Google Colab GPU for full training

## 🚀 Quick Start

### Option 1: Google Colab (Recommended)

1. **Upload to Google Drive**
   - Create folder: `My Drive/TruthLens/`
   - Upload all project files

2. **Open in Google Colab**
   - Navigate to `My Drive/TruthLens/`
   - Open `COLAB_00_quickstart.ipynb`

3. **Run Training**
   - `Runtime → Change runtime type → GPU`
   - Open `COLAB_02_full_training.ipynb`
   - `Runtime → Run all`

4. **Download Results**
   - Model: `artifacts/checkpoints/consistency_aware/consistency_aware_best.pt`
   - Plots: `results/training_results.png`
   - Metrics: `results/consistency_aware_results.json`

### Option 2: VS Code Local

```bash
# Extract ZIP
# Open folder in VS Code

# Setup
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# Fetch a balanced 2,000-example subset and its images
python scripts/fetch_fakeddit_subset.py --per-class 1000

# Create train/validation/test splits
python scripts/prepare_dataset.py --csv data/fakeddit/fakeddit_train_subset.tsv --images data/fakeddit/images

# Train
python scripts/train_model.py --model consistency_aware --epochs 1 --batch-size 4

# Evaluate on the held-out test split
python scripts/evaluate_model.py --checkpoint artifacts/checkpoints/consistency_aware/consistency_aware_best.pt

# Run App
streamlit run app.py
```

### 📥 Download Dataset

1. Visit: https://fakeddit.netlify.app/
2. Download public split
3. Extract images to `data/fakeddit/images/`

Or use sample data in Colab notebooks (no download needed)

## Project Structure

```
multimodal-fake-news/
├── README.md                 # This file
├── requirements.txt          # Local dependencies
├── requirements-colab.txt    # Colab dependencies
├── app.py                    # Streamlit web application
├── configs/
│   └── experiment.yaml       # Experiment configuration
├── src/
│   ├── dataset.py            # Dataset loading and preprocessing
│   ├── preprocessing.py      # Text and image preprocessing
│   ├── encoders.py           # BERT, ViT, and CLIP encoders
│   ├── fusion.py             # Multimodal fusion module
│   ├── classifier.py         # Classification head
│   ├── train.py              # Training pipeline
│   ├── evaluate.py           # Evaluation metrics
│   ├── predict.py            # Inference pipeline
│   ├── image_swap.py         # Image-swap experiment
│   ├── metrics.py            # Metric calculations
│   ├── plotting.py           # Visualization utilities
│   ├── reporting.py          # Report generation
│   └── utils.py              # Utility functions
├── notebooks/
│   ├── 01_environment_check.ipynb
│   ├── 02_dataset_analysis.ipynb
│   ├── 03_baseline_models.ipynb
│   ├── 04_consistency_aware_training.ipynb
│   ├── 05_image_swap_experiment.ipynb
│   └── 06_evaluation_and_plots.ipynb
├── tests/
│   ├── test_dataset.py
│   ├── test_fusion_dimensions.py
│   ├── test_metrics.py
│   └── test_inference.py
├── scripts/
│   ├── prepare_dataset.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── generate_report.py
├── docs/
│   ├── architecture.md
│   ├── datasets.md
│   ├── methodology.md
│   ├── vscode_colab_setup.md
│   └── troubleshooting.md
├── artifacts/                # Cached features and checkpoints
├── results/                  # Experiment results
└── reports/                  # Generated reports
```

## Model Checkpoints

After training, checkpoints are saved to `artifacts/checkpoints/`. Load checkpoints for inference:

```python
from src.predict import NewsAnalyzer
analyzer = NewsAnalyzer(checkpoint_path="artifacts/checkpoints/consistency_aware/consistency_aware_best.pt")
result = analyzer.analyze("News headline here", "path/to/image.jpg")
```

## Web Application Pages

1. **News Analysis**: Submit text and images for fake news detection
2. **Model Performance**: View training metrics and evaluation results
3. **Research Experiments**: Access ablation studies and image-swap results
4. **Documentation**: Learn about the system architecture and methodology

## Research Experiments

### Baseline Models

- **Model A**: Text-only BERT classifier
- **Model B**: Image-only ViT classifier
- **Model C**: Standard multimodal fusion (BERT + ViT)
- **Model D**: Consistency-aware fusion (BERT + ViT + CLIP similarity)
- **Model E**: Ablation without CLIP similarity

### Image-Swap Experiment

Systematically replaces images with unrelated images to test model robustness:

1. Evaluate on original pairs
2. Evaluate on image-swapped pairs
3. Compare prediction probability changes
4. Analyze error cases

## Known Limitations

1. **Dataset Labels**: Fakeddit uses distant supervision - labels may contain noise
2. **Semantic vs. Factual**: CLIP measures semantic similarity, not factual accuracy
3. **Generalization**: Model trained on Reddit data may not generalize to all news sources
4. **Image Quality**: Corrupt or low-quality images affect performance
5. **Text Length**: Maximum 128 tokens for BERT (configurable)

## Citation

If you use this code in your research, please cite:

```bibtex
@software{truthlens_ai,
  title={TruthLens AI: Cross-Modal Consistency-Aware Multimodal Fake News Detection},
  author={Research Team},
  year={2024},
  url={https://github.com/your-repo/multimodal-fake-news}
}
```

## License

This project is for research purposes. Please respect the licenses of:
- Fakeddit Dataset (CC BY 4.0)
- NewsCLIPpings Dataset
- Hugging Face model licenses (BERT, ViT, CLIP)

## Support

For issues and questions, please refer to `docs/troubleshooting.md` or open an issue on the repository.
