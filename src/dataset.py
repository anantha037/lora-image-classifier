import os
from pathlib import Path
from typing import Tuple, List, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
import sys

# Append parent dir for importing Config if necessary
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

class PlantDiseaseDataset(Dataset):
    """
    A PyTorch Dataset class for the PlantVillage disease classification dataset.
    
    Attributes:
        image_paths (List[Path]): A list of paths to images.
        labels (List[int]): A list of integer labels corresponding to the images.
        transform (callable, optional): Optional transform to be applied on a sample.
    """
    
    def __init__(self, image_paths: List[Path], labels: List[int], transform: Optional[callable] = None):
        """
        Initializes the PlantDiseaseDataset.
        
        Args:
            image_paths (List[Path]): List of image file paths.
            labels (List[int]): List of integer labels.
            transform (callable, optional): Transform to be applied to the images.
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.image_paths)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Fetches an image and its corresponding label at a given index.
        
        Args:
            idx (int): Index of the sample.
            
        Returns:
            Tuple[torch.Tensor, int]: Transformed image tensor and its label.
        """
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label


def get_class_names(config: Config) -> List[str]:
    """
    Returns an ordered list of class names present in the dataset root directory.
    If SELECTED_CLASSES is defined in config, it uses that order and intersection.
    Otherwise, it infers classes directly from the subdirectory names.
    
    Args:
        config (Config): Project configuration object.
        
    Returns:
        List[str]: Ordered list of class names.
    """
    dataset_root = Path(config.DATASET_ROOT)
    if not dataset_root.exists():
        # Fallback to selected classes if folder doesn't exist yet (e.g. for testing)
        return getattr(config, 'SELECTED_CLASSES', [])
        
    found_classes = [d.name for d in dataset_root.iterdir() if d.is_dir()]
    
    selected = getattr(config, 'SELECTED_CLASSES', [])
    if selected:
        # Keep only the selected classes that actually exist, preserving Config order
        class_names = [cls for cls in selected if cls in found_classes]
    else:
        class_names = sorted(found_classes)
        
    return class_names


def get_dataloaders(config: Config) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates and returns PyTorch DataLoaders for train, validation, and test splits.
    Performs stratified splitting based on config parameters.
    
    Args:
        config (Config): Project configuration object.
        
    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: Train, validation, and test dataloaders.
    """
    dataset_root = Path(config.DATASET_ROOT)
    class_names = get_class_names(config)
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}
    
    all_image_paths = []
    all_labels = []
    
    # Collect all image paths and labels
    for cls_name in class_names:
        cls_dir = dataset_root / cls_name
        if not cls_dir.exists():
            continue
        for img_path in cls_dir.glob("*.*"):
            if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                all_image_paths.append(img_path)
                all_labels.append(class_to_idx[cls_name])
                
    if not all_image_paths:
        raise ValueError(f"No images found in {dataset_root}. Ensure dataset is downloaded and organized.")
        
    # Stratified Split
    # First split into train and temp (val + test)
    val_test_ratio = config.VAL_SPLIT + config.TEST_SPLIT
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        all_image_paths, all_labels, 
        test_size=val_test_ratio, 
        random_state=config.RANDOM_SEED, 
        stratify=all_labels
    )
    
    # Then split temp into val and test
    test_ratio_of_temp = config.TEST_SPLIT / val_test_ratio
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, 
        test_size=test_ratio_of_temp, 
        random_state=config.RANDOM_SEED, 
        stratify=temp_labels
    )
    
    # ImageNet Mean and Std
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(config.IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])
    
    # Datasets
    train_dataset = PlantDiseaseDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = PlantDiseaseDataset(val_paths, val_labels, transform=val_test_transform)
    test_dataset = PlantDiseaseDataset(test_paths, test_labels, transform=val_test_transform)
    
    # DataLoaders
    # Using num_workers=0 to avoid issues on windows mostly, but let's stick to 0 for POC to be safe, or 4
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=0, 
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=0, 
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=0, 
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
