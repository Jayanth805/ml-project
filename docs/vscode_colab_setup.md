# VS Code + Google Colab Setup Guide

This guide explains how to develop locally with VS Code and train models on Google Colab GPU.

## Option 1: VS Code with Remote SSH (Recommended)

### Prerequisites

1. VS Code installed
2. Remote - SSH extension installed
3. Google Colab account
4. Google Drive access

### Setup Steps

1. **Install VS Code Extensions**
   - Python extension
   - Jupyter extension

2. **Connect to Colab via Remote SSH** (if you have a Colab VM)
   - Use the "Remote SSH" extension to connect to your Colab VM
   - Work directly in VS Code with GPU access

3. **Local Development + Manual Upload**
   - Develop locally in VS Code
   - Upload notebooks to Colab manually
   - Download results back locally

## Option 2: Local + Colab Runtime (No SSH)

### Local Setup

1. **Install Dependencies**
   ```bash
   # Clone or extract project
   cd multimodal-fake-news

   # Create virtual environment
   python -m venv .venv
   .venv\Scripts\activate  # Windows

   # Install dependencies
   pip install -r requirements.txt
   ```

2. **Verify Installation**
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

### Google Colab Setup

1. **Upload Project to Google Drive**
   ```bash
   # Create a folder in Google Drive
   # Upload the project files
   ```

2. **Open Notebook in Colab**
   - Open Google Drive
   - Right-click notebook → Open with → Google Colaboratory

3. **Mount Google Drive**
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

4. **Set Working Directory**
   ```python
   import os
   os.chdir('/content/drive/MyDrive/multimodal-fake-news')
   ```

5. **Install Dependencies in Colab**
   ```bash
   !pip install -r requirements-colab.txt
   ```

6. **Verify GPU Access**
   ```python
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"GPU: {torch.cuda.get_device_name(0)}")
   ```

## GPU Verification in Colab

Run this cell to verify GPU setup:

```python
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
```

Expected output:
```
PyTorch version: 2.0.0
CUDA available: True
CUDA version: 12.1
GPU: Tesla T4
GPU Memory: 14.76 GB
```

## Data Synchronization

### Option A: Google Drive (Recommended)

1. Upload project to Google Drive
2. Upload dataset to Google Drive
3. Mount Drive in Colab
4. Access files directly

### Option B: Direct Upload

1. Upload notebooks to Colab
2. Use Colab file manager for small files
3. Download results after training

### Option C: Git Sync

1. Initialize git in project folder
2. Push to GitHub/Bitbucket
3. Clone in Colab:
   ```bash
   !git clone https://github.com/your-repo/multimodal-fake-news.git
   ```

## Running Training in Colab

### Step-by-Step

1. Open `notebooks/04_consistency_aware_training.ipynb`

2. Update paths in the notebook:
   ```python
   DATA_DIR = '/content/drive/MyDrive/data/processed'
   IMAGES_DIR = '/content/drive/MyDrive/data/fakeddit/images'
   ```

3. Set runtime to GPU:
   - Runtime → Change runtime type → GPU

4. Run cells sequentially

5. Save checkpoints to Google Drive

## Downloading Results

After training in Colab:

```python
# Download a file
from google.colab import files
files.download('results/consistency_aware_results.json')

# Or copy to Drive
!cp results/*.json /content/drive/MyDrive/multimodal-fake-news/results/
```

## Local Inference

After training, download checkpoints locally:

1. Download from Google Drive
2. Place in `artifacts/checkpoints/consistency_aware/`
3. Run Streamlit app locally:
   ```bash
   streamlit run app.py
   ```

## Troubleshooting

### Colab GPU Not Available

- Check Runtime → Change runtime type → GPU
- Try restarting runtime
- May need to request GPU access

### Out of Memory

- Reduce batch size in config
- Use gradient accumulation
- Clear GPU cache:
  ```python
  import torch
  torch.cuda.empty_cache()
  ```

### File Not Found

- Verify paths match exactly
- Check mounted Drive path
- Use absolute paths

## VS Code Extensions Recommended

- Python
- Jupyter
- GitLens
- GitHub Pull Requests
- Remote SSH (if using remote VM)
