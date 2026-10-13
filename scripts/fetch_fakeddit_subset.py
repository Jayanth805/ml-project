"""Fetch a balanced, image-backed Fakeddit subset for local experiments.

The full Fakeddit image archive is very large. This helper samples a manageable
subset from a public Hugging Face mirror and downloads only the selected images.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path

import pandas as pd
import pyarrow.parquet as parquet
import requests
from PIL import Image
from tqdm import tqdm


DATA_URL = (
    "https://huggingface.co/datasets/AdoCleanCode/Fakeddit/resolve/main/"
    "data/train-00000-of-00001.parquet?download=true"
)


def fetch_image(image_id: str, url: str, output_dir: Path) -> bool:
    target = output_dir / f"{image_id}.jpg"
    if target.is_file():
        return True
    try:
        response = requests.get(url, timeout=(8, 20), headers={"User-Agent": "TruthLensResearch/1.0"})
        response.raise_for_status()
        if len(response.content) > 12 * 1024 * 1024:
            return False
        with Image.open(BytesIO(response.content)) as image:
            image.convert("RGB").save(target, format="JPEG", quality=92)
        return True
    except Exception:
        target.unlink(missing_ok=True)
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", default="data/fakeddit/fakeddit-train.parquet")
    parser.add_argument("--output", default="data/fakeddit/fakeddit_train_subset.tsv")
    parser.add_argument("--images", default="data/fakeddit/images")
    parser.add_argument("--per-class", type=int, default=1000,
                        help="Number of true and non-true examples to sample")
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    parquet_path = Path(args.parquet)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    if not parquet_path.exists():
        print(f"Downloading Fakeddit metadata from {DATA_URL}")
        with requests.get(DATA_URL, stream=True, timeout=60) as response:
            response.raise_for_status()
            with parquet_path.open("wb") as output:
                for chunk in response.iter_content(1024 * 1024):
                    if chunk:
                        output.write(chunk)

    table = parquet.read_table(
        parquet_path,
        columns=["id", "text", "6_way_label", "hasImage", "image_url"],
    )
    frame = table.to_pandas()
    frame = frame[frame["hasImage"] & frame["image_url"].notna() & frame["text"].notna()].copy()
    frame["label"] = (frame["6_way_label"] != 0).astype(int)

    available = frame["label"].value_counts().to_dict()
    per_class = min(args.per_class, available.get(0, 0), available.get(1, 0))
    if per_class < 2:
        raise RuntimeError("Not enough labeled image-text examples to create a train/validation split.")
    subset = pd.concat([
        frame[frame["label"] == label].sample(n=per_class, random_state=42)
        for label in (0, 1)
    ], ignore_index=True).sample(frac=1, random_state=42)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = pd.DataFrame({
        "clean_title": subset["text"].astype(str),
        "image_id": subset["id"].astype(str),
        # 0 means true; all five non-true Fakeddit categories are grouped as fake.
        "3_way_label": subset["label"].astype(int),
        "image_url": subset["image_url"].astype(str),
    })
    output.to_csv(output_path, sep="\t", index=False)

    image_dir = Path(args.images)
    image_dir.mkdir(parents=True, exist_ok=True)
    succeeded = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(fetch_image, row.image_id, row.image_url, image_dir)
            for row in output.itertuples(index=False)
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading selected images"):
            succeeded += bool(future.result())

    print(f"Saved {len(output)} balanced examples to {output_path}")
    print(f"Downloaded or already had {succeeded}/{len(output)} selected images in {image_dir}")
    print("Label convention: 0=Fakeddit True; 1=all other Fakeddit categories.")


if __name__ == "__main__":
    main()
