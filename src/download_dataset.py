import os
import shutil
import subprocess
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

def download_and_prepare():
    dataset = "vipoooool/new-plant-diseases-dataset"
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading {dataset}...")
    try:
        subprocess.run(["kaggle", "datasets", "download", "-d", dataset, "-p", str(raw_dir), "--unzip"], check=True)
    except Exception as e:
        print(f"Failed to download dataset. Ensure Kaggle API is configured. Error: {e}")
        return

    # Search for 'train' directory in the unzipped contents
    train_dir = None
    for root, dirs, files in os.walk(raw_dir):
        if "train" in dirs:
            train_dir = Path(root) / "train"
            break
            
    if not train_dir or not train_dir.exists():
        print("Could not find 'train' directory in the downloaded dataset.")
        return

    # Count images per class
    class_counts = {}
    for d in train_dir.iterdir():
        if d.is_dir():
            class_counts[d.name] = len(list(d.glob("*.*")))

    # Select top 10 most common classes
    top_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    top_class_names = [name for name, _ in top_classes]

    print(f"\nTop 10 selected classes:")
    for name, count in top_classes:
        print(f"  {name}: {count} images")

    # Organize them into data/plantvillage/<class_name>/
    target_root = Path(Config.DATASET_ROOT)
    target_root.mkdir(parents=True, exist_ok=True)

    total_images = 0
    for class_name in top_class_names:
        class_target = target_root / class_name
        class_target.mkdir(parents=True, exist_ok=True)
        
        # Combine both train and valid images into one pool (we will split them using dataset.py)
        source_train = train_dir / class_name
        source_valid = train_dir.parent / "valid" / class_name if train_dir.parent.joinpath("valid").exists() else None
        
        sources = [source_train]
        if source_valid and source_valid.exists():
            sources.append(source_valid)
            
        copied_for_class = 0
        for src in sources:
            if not src.exists(): continue
            for img in src.glob("*.*"):
                shutil.copy2(img, class_target / img.name)
                copied_for_class += 1
                
        total_images += copied_for_class
        print(f"Copied {copied_for_class} images for class '{class_name}'")

    print(f"\nTotal images organized: {total_images}")
    print(f"Please ensure config.py SELECTED_CLASSES has these classes: {top_class_names}")

if __name__ == "__main__":
    download_and_prepare()
