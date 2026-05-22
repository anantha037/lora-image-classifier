# Run this file after training is complete and adapter weights are saved.
# Expects: outputs/checkpoints/ to contain saved LoRA adapter weights.

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
from pathlib import Path

# Append parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from src.dataset import get_dataloaders, get_class_names
from src.model import load_adapter

def evaluate_model(model, test_loader, device, class_names):
    """
    Runs inference over the full test dataloader and computes accuracy metrics.
    
    Args:
        model: The trained PEFT model to evaluate.
        test_loader: DataLoader for the test set.
        device: The device to run inference on.
        class_names: List of ordered class names.
        
    Returns:
        tuple: (all_preds, all_labels) as flat Python lists.
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    print("\nRunning inference on test set...")
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # Compute accuracy
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    overall_acc = (all_preds == all_labels).mean() * 100
    
    print(f"\nOverall Test Accuracy: {overall_acc:.2f}%\n")
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))
    
    return all_preds.tolist(), all_labels.tolist()

def plot_confusion_matrix(all_preds, all_labels, class_names, save_path):
    """
    Computes and plots a confusion matrix.
    
    Args:
        all_preds: Flat list of predicted labels.
        all_labels: Flat list of ground truth labels.
        class_names: List of ordered class names.
        save_path: Path to save the output plot.
        
    Returns:
        np.ndarray: The computed confusion matrix.
    """
    cm = confusion_matrix(all_labels, all_preds)
    # Normalize per row (recall) for color mapping
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.matshow(cm_normalized, cmap='Blues')
    fig.colorbar(cax)
    
    # Annotate cells
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), va='center', ha='center',
                    color="white" if cm_normalized[i, j] > 0.5 else "black")
            
    # Set axis labels and ticks
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="left")
    ax.set_yticklabels(class_names)
    ax.xaxis.tick_bottom()
    
    plt.title('Confusion Matrix', pad=20)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    plt.savefig(save_path)
    print(f"Confusion matrix saved to {save_path}")
    plt.close()
    
    return cm

def _denormalize(tensor, mean, std):
    """
    Unnormalizes an image tensor.
    
    Args:
        tensor: Image tensor of shape (C, H, W).
        mean: List of mean values used for normalization.
        std: List of std values used for normalization.
        
    Returns:
        np.ndarray: Denormalized image array ready for matplotlib display.
    """
    tensor = tensor.clone().detach().cpu()
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    tensor = tensor * std + mean
    tensor = torch.clamp(tensor, 0, 1)
    return tensor.permute(1, 2, 0).numpy()

def plot_misclassified(model, test_loader, device, class_names, n=16):
    """
    Finds and plots the first n misclassified samples.
    
    Args:
        model: The trained PEFT model.
        test_loader: DataLoader for the test set.
        device: The device to run inference on.
        class_names: List of ordered class names.
        n: Number of misclassified images to plot.
    """
    model.eval()
    misclassified_images = []
    misclassified_preds = []
    misclassified_trues = []
    
    # Standard ImageNet mean and std as used in dataset.py
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    
    with torch.no_grad():
        for images, labels in test_loader:
            images_device = images.to(device)
            outputs = model(images_device)
            _, preds = torch.max(outputs, 1)
            
            # Find misclassified indices in batch
            wrong_idx = (preds.cpu() != labels).nonzero(as_tuple=True)[0]
            
            for idx in wrong_idx:
                if len(misclassified_images) < n:
                    misclassified_images.append(images[idx])
                    misclassified_preds.append(preds[idx].item())
                    misclassified_trues.append(labels[idx].item())
                else:
                    break
            if len(misclassified_images) >= n:
                break
                
    if not misclassified_images:
        print("No misclassified samples found! (100% accuracy?)")
        return
        
    # Plotting
    cols = 4
    rows = int(np.ceil(len(misclassified_images) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
    axes = axes.flatten()
    
    for i in range(len(misclassified_images)):
        img_np = _denormalize(misclassified_images[i], imagenet_mean, imagenet_std)
        true_name = class_names[misclassified_trues[i]]
        pred_name = class_names[misclassified_preds[i]]
        
        ax = axes[i]
        ax.imshow(img_np)
        ax.axis('off')
        
        # Format title to show True vs Pred. Color wrong pred red.
        title_text = f"True: {true_name}\nPred: {pred_name}"
        ax.set_title(title_text, color='red', fontsize=9)
        
    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout()
    
    save_path = "outputs/misclassified_samples.png"
    plt.savefig(save_path)
    print(f"Misclassified samples saved to {save_path}")
    plt.close()

def run_evaluation(config: Config):
    """
    Main evaluation pipeline that loads data, model adapter, evaluates,
    and plots results.
    """
    print("Initializing Evaluation Pipeline...")
    device = torch.device(config.DEVICE)
    class_names = get_class_names(config)
    
    # Load test dataloader
    _, _, test_loader = get_dataloaders(config)
    
    # Load model adapter
    adapter_path = config.CHECKPOINT_DIR
    if not os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
        print(f"ERROR: Adapter not found in {adapter_path}. Ensure training is complete.")
        return
        
    model = load_adapter(config, adapter_path)
    print(f"Loaded trained adapter from {adapter_path}")
    
    # Evaluate
    all_preds, all_labels = evaluate_model(model, test_loader, device, class_names)
    
    # Confusion Matrix
    cm_path = "outputs/confusion_matrix.png"
    plot_confusion_matrix(all_preds, all_labels, class_names, cm_path)
    
    # Misclassified
    plot_misclassified(model, test_loader, device, class_names, n=16)
    
    # Final Summary
    overall_acc = (np.array(all_preds) == np.array(all_labels)).mean() * 100
    
    print("\n" + "="*30)
    print("EVALUATION SUMMARY")
    print("="*30)
    print(f"Test Accuracy     : {overall_acc:.2f}%")
    print(f"Confusion Matrix  : {cm_path}")
    print(f"Misclassified     : outputs/misclassified_samples.png")
    print("="*30 + "\n")

if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    config = Config()
    run_evaluation(config)
