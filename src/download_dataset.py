import os
import shutil
import subprocess
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

SELECTED_CLASSES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Corn___Common_rust",
    "Grape___Black_rot",
    "Grape___healthy",
    "Potato___Early_blight",
    "Tomato___Bacterial_spot",
    "Tomato___healthy"
]

def download_and_prepare():
    dataset = "vipoooool/new-plant-diseases-dataset"
    raw_dir = Path("data/raw")
    target_root = Path(Config.DATASET_ROOT)
    
    # Check if dataset already has the classes populated
    if target_root.exists():
        all_exist = True
        for class_name in SELECTED_CLASSES:
            class_dir = target_root / class_name
            if not class_dir.exists() or len(list(class_dir.glob("*.*"))) == 0:
                all_exist = False
                break
        if all_exist:
            print("Dataset already populated with selected classes. Skipping download.")
            return

    raw_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading {dataset}...")
    try:
        subprocess.run(["kaggle", "datasets", "download", "-d", dataset, "-p", str(raw_dir), "--unzip"], check=True)
    except Exception as e:
        print(f"Failed to download dataset. Ensure Kaggle API is configured. Error: {e}")
        return

    # Search for 'train' and 'valid' directories in the unzipped contents
    train_dir = None
    valid_dir = None
    for root, dirs, files in os.walk(raw_dir):
        if "train" in dirs:
            train_dir = Path(root) / "train"
        if "valid" in dirs:
            valid_dir = Path(root) / "valid"
            
    if not train_dir or not train_dir.exists():
        print("Could not find 'train' directory in the downloaded dataset.")
        return
    if not valid_dir or not valid_dir.exists():
        print("Could not find 'valid' directory in the downloaded dataset.")

    target_root.mkdir(parents=True, exist_ok=True)

    total_images = 0
    print("\nOrganizing selected classes:")
    for class_name in SELECTED_CLASSES:
        class_target = target_root / class_name
        class_target.mkdir(parents=True, exist_ok=True)
        
        # Find matching folder in train_dir
        train_match = None
        if (train_dir / class_name).exists():
            train_match = train_dir / class_name
        else:
            # Fuzzy match
            class_keywords = class_name.lower().replace("___", " ").replace("_", " ").split()
            for d in train_dir.iterdir():
                if d.is_dir():
                    d_name_lower = d.name.lower().replace("___", " ").replace("_", " ")
                    if all(k in d_name_lower for k in class_keywords):
                        train_match = d
                        break
                        
        sources = []
        if train_match:
            sources.append(train_match)
            # Try to find corresponding valid folder
            if valid_dir and valid_dir.exists():
                valid_match = valid_dir / train_match.name
                if valid_match.exists():
                    sources.append(valid_match)
        else:
            print(f"  Warning: Could not find matching folder for {class_name} in dataset.")
            continue

        copied_for_class = 0
        for src in sources:
            for img in src.glob("*.*"):
                shutil.copy2(img, class_target / img.name)
                copied_for_class += 1
                
        total_images += copied_for_class
        print(f"  {class_name}: {copied_for_class} images")

    print(f"\nTotal images organized: {total_images}")

if __name__ == "__main__":
    download_and_prepare()
