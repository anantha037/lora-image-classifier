# Grad-CAM for Vision Transformers requires a reshape_transform
# because ViT outputs token sequences, not spatial feature maps.
# We skip the CLS token and reshape patch tokens into a 2D spatial grid.

import os
import sys
import random
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# Append parent dir for importing
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from src.model import load_adapter
from src.dataset import get_class_names


def reshape_transform(tensor, height=14, width=14):
    """
    Reshape transform for ViT.
    The input tensor has shape [batch_size, num_tokens, dim].
    We skip the first token (CLS token), and reshape the remaining patch tokens
    into a 2D spatial grid [batch_size, dim, height, width].
    """
    # Skip CLS token and reshape to [batch, height, width, dim]
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    # Transpose to [batch, dim, height, width]
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def generate_gradcam(model, image_tensor, class_idx, device):
    """
    Generates Grad-CAM heatmap for a given model, image tensor, and target class.
    
    Args:
        model (PeftModel): The LoRA-tuned ViT model.
        image_tensor (torch.Tensor): Preprocessed image tensor [1, 3, H, W].
        class_idx (int): The target class index for Grad-CAM.
        device (str/torch.device): Device to run Grad-CAM on.
        
    Returns:
        np.ndarray: Raw CAM numpy array (H x W float).
    """
    # Unwrap PEFT to access base ViT
    base_model = model.base_model.model
    
    # Target the last attention block's norm1 for Grad-CAM
    target_layer = base_model.blocks[-1].norm1
    
    # Initialize GradCAM
    cam = GradCAM(
        model=model, 
        target_layers=[target_layer], 
        reshape_transform=reshape_transform, 
    )
    
    # Generate CAM
    targets = [ClassifierOutputTarget(class_idx)]
    grayscale_cam = cam(input_tensor=image_tensor, targets=targets)
    
    # For a single image batch, return the first item
    return grayscale_cam[0, :]


def overlay_gradcam(original_image_path, cam, save_path, pred_class="", confidence=0.0):
    """
    Visualizes the Grad-CAM heatmap over the original image and saves it.
    
    Args:
        original_image_path (str): Path to the original image.
        cam (np.ndarray): Raw CAM heatmap array.
        save_path (str): Where to save the resulting side-by-side visualization.
        pred_class (str): Predicted class name for the title.
        confidence (float): Confidence percentage for the title.
        
    Returns:
        np.ndarray: Heatmap numpy array (overlay image).
    """
    # Open and resize the original image
    img = Image.open(original_image_path).convert("RGB")
    img = img.resize((224, 224))
    
    # Convert image to [0, 1] float array for show_cam_on_image
    img_float = np.float32(img) / 255.0
    
    # Overlay CAM on image
    heatmap = show_cam_on_image(img_float, cam, use_rgb=True)
    
    # Create side-by-side figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    axes[0].imshow(img)
    axes[0].set_title("Original")
    axes[0].axis("off")
    
    axes[1].imshow(heatmap)
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis("off")
    
    # Add main title with prediction and confidence
    fig.suptitle(f"Predicted: {pred_class} ({confidence:.2f}%)", fontsize=16)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    return heatmap


def explain_batch(model, image_paths, config, class_names):
    """
    Processes a list of image paths, runs inference, generates Grad-CAM overlays, 
    and saves them.
    
    Args:
        model (PeftModel): The trained model.
        image_paths (List[str]): List of paths to images.
        config (Config): Configuration object.
        class_names (List[str]): List of class names.
        
    Returns:
        List[dict]: Results containing image path, prediction, confidence, and saved cam path.
    """
    # Val/Test transforms (must match dataset.py)
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    os.makedirs(config.GRADCAM_OUTPUT_DIR, exist_ok=True)
    
    results = []
    
    for path in image_paths:
        try:
            # 1. Load and preprocess image
            img = Image.open(path).convert("RGB")
            tensor = val_transform(img).unsqueeze(0).to(config.DEVICE)
            
            # 2. Forward pass to get prediction
            with torch.no_grad():
                outputs = model(tensor)
                probs = torch.softmax(outputs, dim=1)
                conf, pred_idx = torch.max(probs, dim=1)
                
            conf = conf.item() * 100.0
            pred_idx = pred_idx.item()
            pred_class = class_names[pred_idx]
            
            # 3. Generate Grad-CAM
            cam = generate_gradcam(model, tensor, pred_idx, config.DEVICE)
            
            # 4. Save overlay
            filename = os.path.basename(path)
            save_path = os.path.join(config.GRADCAM_OUTPUT_DIR, f"{filename}_gradcam.png")
            
            overlay_gradcam(path, cam, save_path, pred_class, conf)
            
            print(f"Image: {filename} | Predicted: {pred_class} | Confidence: {conf:.2f}%")
            
            results.append({
                "image": path,
                "pred_class": pred_class,
                "confidence": conf,
                "cam_path": save_path
            })
            
        except Exception as e:
            print(f"Failed to process image {path}: {str(e)}")
            continue
            
    return results


def run_gradcam_demo(config):
    """
    End-to-end demo that loads the model, selects random images from each class,
    generates Grad-CAM explanations, and compiles a summary grid.
    
    Args:
        config (Config): Configuration object.
    """
    print("Initializing Grad-CAM Explainability Demo...")
    
    # Load model and class names
    model = load_adapter(config, config.CHECKPOINT_DIR)
    class_names = get_class_names(config)
    
    # Find sample images (2 from each class)
    dataset_root = Path(config.DATASET_ROOT)
    sample_paths = []
    
    for cls in class_names:
        cls_dir = dataset_root / cls
        if cls_dir.exists():
            imgs = [p for p in cls_dir.glob("*.*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]
            if len(imgs) >= 2:
                selected = random.sample(imgs, 2)
            else:
                selected = imgs
            sample_paths.extend([str(p) for p in selected])
            
    if not sample_paths:
        print("No images found for Grad-CAM demo. Please check your dataset.")
        return
        
    print(f"Selected {len(sample_paths)} images for Grad-CAM processing.")
    
    # Run explain batch
    results = explain_batch(model, sample_paths, config, class_names)
    
    # Create summary grid
    if results:
        print("\nCreating Grad-CAM summary grid...")
        # 4 rows x 5 columns grid layout
        rows, cols = 4, 5
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
        axes = axes.flatten()
        
        for idx in range(len(axes)):
            if idx < len(results):
                res = results[idx]
                img = Image.open(res["cam_path"])
                axes[idx].imshow(img)
                title = f"Pred: {res['pred_class']}"
                axes[idx].set_title(title, fontsize=9)
            axes[idx].axis("off")
            
        plt.tight_layout()
        grid_path = "outputs/gradcam_summary_grid.png"
        os.makedirs("outputs", exist_ok=True)
        plt.savefig(grid_path, dpi=150)
        plt.close()
        print(f"Summary grid saved to {grid_path}")
        
    print("Grad-CAM Explainability Demo Completed Successfully.")


if __name__ == "__main__":
    config = Config()
    run_gradcam_demo(config)
