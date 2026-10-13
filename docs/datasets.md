# Dataset Documentation

## Fakeddit Dataset

### Overview

Fakeddit is a multimodal fake news dataset collected from Reddit. It contains posts with both text and images, along with labels indicating reliability.

### Official Resources

- **Website**: https://fakeddit.netlify.app/
- **GitHub**: https://github.com/entitize/Fakeddit
- **Paper**: Nakamura et al., "Fakeddit: A Multimodal Dataset for Fake News Classification" (LREC 2020)

### Dataset Structure

The Fakeddit dataset includes:

- `clean_title`: Cleaned post title (used as text input)
- `image_id`: Identifier for the associated image
- `label`: Category label
- `2_way_label`: Binary label (reliable/unreliable)
- `3_way_label`: Three-way label (reliable/unreliable/misleading)
- `subreddit`: Source subreddit
- `created_utc`: Post timestamp

### Label Definitions

Fakeddit uses **distant supervision** for labeling based on subreddit membership:

| Label | Description |
|-------|-------------|
| 0 | Reliable sources (from trusted subreddits) |
| 1 | Unreliable sources (from untrusted subreddits) |
| 2 | Misleading content |

### Binary Classification Mapping

For fake news detection, we map to binary labels:

| Original Label | Binary Label | Meaning |
|----------------|--------------|---------|
| 0 (reliable) | 0 | REAL |
| 1 (unreliable) | 1 | FAKE |
| 2 (misleading) | 1 | FAKE |

### Data Download

1. Visit https://fakeddit.netlify.app/
2. Follow the download instructions
3. Download the public split or request access
4. Extract to a local directory

### Directory Structure

```
data/fakeddit/
├── your_fakeddit_file.csv
└── images/
    ├── 001.jpg
    ├── 002.jpg
    └── ...
```

### Limitations

1. **Distant Supervision**: Labels are based on subreddit trust signals, not fact-checking
2. **Reddit Domain**: Model trained on Reddit may not generalize to formal news
3. **Label Noise**: Some labels may be incorrect
4. **Image Availability**: Some image IDs may not have corresponding images

## NewsCLIPpings Dataset

### Overview

NewsCLIPpings is designed for out-of-context image detection. It contains news captions with images that may or may not match the caption context.

### Official Resources

- **Paper**: Luo et al., "NewsCLIPpings: Automatic Detection of News Photographs with Out-of-Context Image Captions" (EMNLP 2021)
- **GitHub**: https://github.com/g-luo/news_clippings

### Task Difference

NewsCLIPpings focuses on **out-of-context detection** (OCC), which is different from fake news detection:

| Task | Goal |
|------|------|
| Fake News Detection | Is the news claim true or false? |
| Out-of-Context | Does the image match the caption? |

Note: These are related but distinct tasks. A mismatched image doesn't necessarily mean fake news.

## Dataset Configuration

Update `configs/experiment.yaml` with your dataset paths:

```yaml
dataset:
  path: "data/fakeddit"
  text_column: "clean_title"
  image_column: "image_id"
  label_column: "3_way_label"
  label_mapping:
    0: 0  # reliable -> real
    1: 1  # unreliable -> fake
    2: 1  # misleading -> fake
```

## Data Preparation

For a local experiment without downloading Fakeddit's full 106 GB image archive,
the repository includes a helper that samples a balanced subset from a public
Hugging Face mirror and downloads only those images:

```powershell
python scripts/fetch_fakeddit_subset.py --per-class 1000
python scripts/prepare_dataset.py --csv data/fakeddit/fakeddit_train_subset.tsv --images data/fakeddit/images
```

This maps Fakeddit's `True` category to class 0 and all other categories to class 1.
The sampled set is intended for local experimentation, not benchmark reporting.

Use the provided script to prepare the dataset:

```bash
python scripts/prepare_dataset.py \
    --csv data/fakeddit/your_file.tsv \
    --images data/fakeddit/images \
    --output data/processed
```

This will:
1. Load and validate the CSV
2. Apply label mapping
3. Create train/val/test splits (70/15/15)
4. Save processed data to `data/processed/`
