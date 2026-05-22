# ============================================================
# TRAINING NOTE: Run this file in Google Colab for GPU support
# Steps:
#   1. Clone repo: !git clone <your-repo-url>
#   2. Install deps: !pip install -r requirements.txt
#   3. Download dataset: !python src/download_dataset.py
#   4. Run training: !python src/train.py
#   5. Save adapter: outputs/checkpoints/ -> copy to Drive
# ============================================================

import os
import sys
import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path

# Append parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from src.dataset import get_dataloaders
from src.model import get_model, save_adapter

def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    """
    Trains the model for one epoch over the given dataloader.
    
    Args:
        model (nn.Module): The PyTorch model to train.
        loader (DataLoader): Training dataloader.
        optimizer (Optimizer): Optimizer for parameter updates.
        criterion (Loss): Loss function.
        device (torch.device): Device to run training on (CPU/GPU).
        scaler (GradScaler): Automatic Mixed Precision scaler.
        
    Returns:
        tuple: (average_loss, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Training", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision context
        if device.type == "cuda":
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            
        # Backward pass and optimize
        if device.type == "cuda":
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
            
        running_loss += loss.item() * images.size(0)
        
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{100.*correct/total:.2f}%"})
        
    avg_loss = running_loss / total
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def validate(model, loader, criterion, device):
    """
    Validates the model on the given dataloader.
    
    Args:
        model (nn.Module): The PyTorch model.
        loader (DataLoader): Validation dataloader.
        criterion (Loss): Loss function.
        device (torch.device): Device to run evaluation on.
        
    Returns:
        tuple: (average_loss, accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Validating", leave=False)
    with torch.no_grad():
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            # Using autocast for validation speedup if on CUDA
            if device.type == "cuda":
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                
            running_loss += loss.item() * images.size(0)
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'val_loss': f"{loss.item():.4f}", 'val_acc': f"{100.*correct/total:.2f}%"})
            
    avg_loss = running_loss / total
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def plot_training_curves(history: dict, config: Config):
    """
    Plots train vs val loss and accuracy curves side by side and saves them.
    
    Args:
        history (dict): Dictionary containing the metrics.
        config (Config): Configuration object for output directory.
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    ax1.plot(epochs, history["train_loss"], label='Train Loss', marker='o')
    ax1.plot(epochs, history["val_loss"], label='Validation Loss', marker='o')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(epochs, history["train_acc"], label='Train Accuracy', marker='o')
    ax2.plot(epochs, history["val_acc"], label='Validation Accuracy', marker='o')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    # Ensure output dir exists
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_path = output_dir / "training_curves.png"
    plt.savefig(plot_path)
    print(f"Training curves saved to {plot_path}")
    plt.close()


def train(config: Config):
    """
    Full training loop combining train and validation steps, with Early Stopping,
    Cosine Annealing LR, and Mixed Precision.
    """
    # Reproducibility
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.RANDOM_SEED)
        
    device = torch.device(config.DEVICE)
    print(f"Starting training on device: {device}")
    
    # Dataloaders and Model
    train_loader, val_loader, _ = get_dataloaders(config)
    model = get_model(config)
    
    # Optimizer (only on trainable params)
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=config.LEARNING_RATE, weight_decay=0.01)
    
    # Scheduler & Loss
    scheduler = CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS)
    criterion = nn.CrossEntropyLoss()
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    
    # Early stopping setup
    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience_counter = 0
    patience = 4
    best_epoch = 0
    
    # Metrics tracking
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    
    print("\nStarting Training...")
    print(f"{'Epoch':<6} | {'Train Loss':<10} | {'Train Acc':<10} | {'Val Loss':<10} | {'Val Acc':<10} | {'LR'}")
    print("-" * 75)
    
    start_time = time.time()
    
    for epoch in range(1, config.NUM_EPOCHS + 1):
        # Train & Validate
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Step scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        # Print metrics
        print(f"{epoch:<6} | {train_loss:<10.4f} | {train_acc:<9.2f}% | {val_loss:<10.4f} | {val_acc:<9.2f}% | {current_lr:.2e}")
        
        # Early Stopping and Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        # Save if accuracy improved
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            save_adapter(model, config.CHECKPOINT_DIR)
            print(f" -> New best model saved! (Val Acc: {best_val_acc:.2f}%)")
            
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch}! No improvement in validation loss for {patience} epochs.")
            break

    total_time = time.time() - start_time
    hours, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)
    
    print("\n" + "="*50)
    print("Training Complete!")
    print(f"Total Time: {int(hours):02d}h {int(minutes):02d}m {int(seconds):02d}s")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}% (Achieved at Epoch {best_epoch})")
    print("="*50 + "\n")
    
    # Plotting metrics
    plot_training_curves(history, config)
    
    return history


if __name__ == "__main__":
    # Create required output directories if they don't exist
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.GRADCAM_OUTPUT_DIR, exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    
    config = Config()
    
    print("Initializing Training Pipeline...")
    train(config)
    print("Training Script Finished.")
