#!/usr/bin/env python3
"""
Download and extract the TensorFlow Flowers dataset.
"""

import os
import urllib.request
import tarfile
from pathlib import Path

DATASET_URL = "http://download.tensorflow.org/example_images/flower_photos.tgz"
DATA_DIR = Path(__file__).parent.parent / "data"
ARCHIVE_PATH = DATA_DIR / "flower_photos.tgz"
EXTRACT_DIR = DATA_DIR / "flower_photos"


def download_dataset():
    """Download the flowers dataset if not already present."""
    
    # Create data directory if needed
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check if already extracted
    if EXTRACT_DIR.exists():
        print(f"✓ Dataset already exists at {EXTRACT_DIR}")
        return
    
    # Download if archive doesn't exist
    if not ARCHIVE_PATH.exists():
        print(f"Downloading dataset from {DATASET_URL}...")
        urllib.request.urlretrieve(DATASET_URL, ARCHIVE_PATH)
        print(f"✓ Downloaded to {ARCHIVE_PATH}")
    else:
        print(f"✓ Archive already exists at {ARCHIVE_PATH}")
    
    # Extract
    print("Extracting dataset...")
    with tarfile.open(ARCHIVE_PATH, "r:gz") as tar:
        tar.extractall(path=DATA_DIR)
    print(f"✓ Extracted to {EXTRACT_DIR}")
    
    # Show dataset info
    show_dataset_info()


def show_dataset_info():
    """Display information about the downloaded dataset."""
    
    if not EXTRACT_DIR.exists():
        print("Dataset not found!")
        return
    
    print("\n" + "=" * 50)
    print("DATASET INFORMATION")
    print("=" * 50)
    
    classes = [d for d in EXTRACT_DIR.iterdir() if d.is_dir()]
    
    total_images = 0
    for class_dir in sorted(classes):
        images = list(class_dir.glob("*.jpg"))
        count = len(images)
        total_images += count
        print(f"  {class_dir.name:15} : {count:4} images")
    
    print("-" * 50)
    print(f"  {'TOTAL':15} : {total_images:4} images")
    print(f"  Classes: {len(classes)}")
    print("=" * 50)


if __name__ == "__main__":
    download_dataset()
