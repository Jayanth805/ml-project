# Troubleshooting Guide

## Common Issues and Solutions

## Installation Issues

### Problem: `ModuleNotFoundError` when importing

**Solution:**
```bash
pip install -r requirements.txt
```

If still failing, install specific packages:
```bash
pip install torch transformers datasets
```

### Problem: CUDA out of memory

**Solutions:**
1. Reduce batch size:
   ```python
   batch_size = 8  # or 4
   ```

2. Use gradient accumulation:
   ```python
   gradient_accumulation_steps = 2
   ```

3. Clear GPU cache:
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

4. Try Google Colab with T4 GPU

### Problem: Transformers model download fails

**Solution:**
```python
from transformers import AutoModel, AutoTokenizer

# Download model explicitly
model_name = "google-bert/bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
```

## Dataset Issues

### Problem: CSV file not found

**Solution:**
Update the path in `configs/experiment.yaml` or pass it as argument:
```bash
python scripts/prepare_dataset.py --csv data/fakeddit/your_file.csv --images data/fakeddit/images
```

### Problem: Images not loading

**Solutions:**
1. Verify image directory path
2. Check file extensions (jpg vs jpeg vs png)
3. Check image files are not corrupted

### Problem: Label mapping errors

**Solution:**
Verify column names match your CSV:
```python
# Check available columns
df.columns.tolist()
```

## Training Issues

### Problem: Training very slow

**Solutions:**
1. Use GPU instead of CPU
2. Enable feature caching
3. Reduce dataset size for testing
4. Use mixed precision:
   ```python
   config.use_amp = True
   ```

### Problem: Model not improving

**Solutions:**
1. Check learning rate (try 1e-4 to 1e-3)
2. Verify labels are correct
3. Check data split (no leakage)
4. Increase training epochs

### Problem: Early stopping triggered too early

**Solutions:**
1. Increase patience:
   ```yaml
   early_stopping_patience: 5
   ```
2. Use different metric (val_loss instead of val_f1)

## Model Issues

### Problem: Checkpoint loading fails

**Solutions:**
1. Verify checkpoint file exists:
   ```bash
   ls artifacts/checkpoints/
   ```
2. Check model config matches checkpoint
3. Try loading with `torch.load(checkpoint, map_location='cpu')`

### Problem: Predictions always same class

**Solutions:**
1. Model not trained - train first
2. Check threshold (try 0.5)
3. Verify input preprocessing

## Streamlit App Issues

### Problem: App not loading

**Solutions:**
1. Check port availability
2. Try different port:
   ```bash
   streamlit run app.py --server.port 8502
   ```
3. Clear browser cache

### Problem: "No trained model loaded"

**Solution:**
Train model first:
```bash
python scripts/train_model.py --model consistency_aware --epochs 5
```

### Problem: Image upload fails

**Solutions:**
1. Check file size (max ~10MB)
2. Supported formats: jpg, jpeg, png
3. Check browser console for errors

## Jupyter Notebook Issues

### Problem: Notebook kernel dying

**Solutions:**
1. Reduce batch size
2. Clear outputs and re-run
3. Restart kernel

### Problem: Model download in notebook slow

**Solution:**
Download models once and cache:
```python
from transformers import AutoModel
model = AutoModel.from_pretrained("google-bert/bert-base-uncased")
```

## Google Colab Issues

### Problem: Runtime disconnecting

**Solutions:**
1. Keep browser tab open
2. Use Chrome instead of Firefox
3. Install Colab-specific packages:
   ```bash
   !pip install -r requirements-colab.txt
   ```

### Problem: GPU not available

**Solution:**
Runtime → Change runtime type → GPU

### Problem: Mount Drive fails

**Solution:**
```python
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
```

## Getting Help

If you encounter an issue not covered here:

1. Check the documentation in `docs/`
2. Review Jupyter notebooks for examples
3. Check GitHub issues (if available)
4. Verify all dependencies are installed

## Error Messages

### "CUDA error: out of memory"

Reduce batch size or use CPU for testing.

### "RuntimeError: Expected tensor for argument #1..."

Check that inputs are on correct device (cuda/cpu).

### "ValueError: too many values to unpack"

Check model output shapes and tuple unpacking.

### "FileNotFoundError: No such file or directory"

Verify all paths are correct and files exist.
