"""
Script to prepare and verify the dataset.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import load_config, ensure_dir, save_json


def prepare_dataset(
    csv_path: str,
    images_dir: str,
    output_dir: str = "data/processed",
    max_samples: int = None
):
    """
    Prepare dataset for training.

    Args:
        csv_path: Path to the dataset CSV file
        images_dir: Path to the images directory
        output_dir: Output directory for processed data
        max_samples: Maximum number of samples to process
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from src.utils import set_seed

    print("=" * 60)
    print("Dataset Preparation Script")
    print("=" * 60)

    # Load config
    config_path = Path(__file__).parent.parent / "configs" / "experiment.yaml"
    if config_path.exists():
        config = load_config(str(config_path))
        print(f"Loaded config from {config_path}")
    else:
        config = {}
        print("No config file found, using defaults")

    # Load dataset
    print(f"\nLoading dataset from {csv_path}...")
    # Fakeddit releases TSV files; CSV is also accepted for custom datasets.
    df = pd.read_csv(csv_path, sep=None, engine='python')
    print(f"Loaded {len(df)} samples")

    if max_samples and max_samples < len(df):
        df = df.head(max_samples)
        print(f"Limited to {max_samples} samples")

    # Identify columns
    text_col = config.get('dataset', {}).get('text_column', 'clean_title')
    image_col = config.get('dataset', {}).get('image_column', 'image_id')
    label_col = config.get('dataset', {}).get('label_column', 'label')

    print(f"\nColumns detected:")
    print(f"  Text: {text_col}")
    print(f"  Image: {image_col}")
    print(f"  Label: {label_col}")

    # Check columns exist
    if text_col not in df.columns:
        print(f"Warning: Column '{text_col}' not found. Available columns: {df.columns.tolist()}")
        # Try to find similar columns
        for col in ['title', 'text', 'content']:
            if col in df.columns:
                text_col = col
                print(f"  Using '{col}' as text column instead")
                break
    if text_col not in df.columns:
        raise ValueError(f"No text column found. Available columns: {df.columns.tolist()}")
    if image_col not in df.columns:
        raise ValueError(f"Image column '{image_col}' not found. Available columns: {df.columns.tolist()}")
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found. Available columns: {df.columns.tolist()}")

    # Apply label mapping
    label_mapping = config.get('dataset', {}).get('label_mapping', {0: 0, 1: 1, 2: 1})
    # YAML may load numeric keys as integers while CSV values arrive as strings.
    normalized_mapping = {str(key).strip().lower(): value for key, value in label_mapping.items()}
    label_values = df[label_col]
    df['binary_label'] = label_values.map(
        lambda value: normalized_mapping.get(str(value).strip().lower()) if pd.notna(value) else None
    )
    df = df[df['binary_label'].notna()].copy()
    print(f"\nAfter label mapping: {len(df)} samples")

    # Check images exist
    print("\nChecking image availability...")
    images_path = Path(images_dir)
    df['image_exists'] = df[image_col].apply(
        lambda image_id: (images_path / f"{image_id}.jpg").is_file() if pd.notna(image_id) else False
    )
    existing = df['image_exists'].sum()
    print(f"Images found: {existing}/{len(df)}")

    # Create output directory
    output_path = Path(output_dir)
    ensure_dir(output_path)

    # Save processed data
    process_df = df[[text_col, image_col, 'binary_label', 'image_exists']].copy()
    process_df.columns = ['text', 'image_id', 'label', 'image_exists']
    process_df.to_csv(output_path / 'fakeddit_processed.csv', index=False)
    print(f"\nSaved processed data to {output_path / 'fakeddit_processed.csv'}")

    # Create splits
    print("\nCreating train/val/test splits...")
    set_seed(42)

    # Split
    train_df, temp_df = train_test_split(
        process_df,
        test_size=0.3,
        random_state=42,
        stratify=process_df['label']
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=42,
        stratify=temp_df['label']
    )

    print(f"  Train: {len(train_df)} ({len(train_df)/len(process_df)*100:.1f}%)")
    print(f"  Val: {len(val_df)} ({len(val_df)/len(process_df)*100:.1f}%)")
    print(f"  Test: {len(test_df)} ({len(test_df)/len(process_df)*100:.1f}%)")

    # Save splits
    train_df.to_csv(output_path / 'train.csv', index=False)
    val_df.to_csv(output_path / 'val.csv', index=False)
    test_df.to_csv(output_path / 'test.csv', index=False)

    print(f"\nSaved splits to {output_path}/")

    # Save summary
    summary = {
        'total_samples': len(process_df),
        'train_samples': len(train_df),
        'val_samples': len(val_df),
        'test_samples': len(test_df),
        'class_distribution': {
            'train': train_df['label'].value_counts().to_dict(),
            'val': val_df['label'].value_counts().to_dict(),
            'test': test_df['label'].value_counts().to_dict()
        },
        'images_available': int(existing)
    }

    save_json(summary, output_path / 'dataset_summary.json')

    print("\n" + "=" * 60)
    print("Dataset preparation complete!")
    print("=" * 60)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for training")
    parser.add_argument("--csv", required=True, help="Path to dataset CSV file")
    parser.add_argument("--images", required=True, help="Path to images directory")
    parser.add_argument("--output", default="data/processed", help="Output directory")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum samples to process")

    args = parser.parse_args()

    prepare_dataset(args.csv, args.images, args.output, args.max_samples)


if __name__ == "__main__":
    main()
