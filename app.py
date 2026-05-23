# Run with: python app.py
# Demo at:  http://localhost:7860

import os
import torch
import gradio as gr
from PIL import Image
from torchvision import transforms

from config import Config
from src.model import load_adapter
from src.dataset import get_class_names
from src.gradcam import generate_gradcam, overlay_gradcam

print("Loading configuration and model...")
config = Config()
class_names = config.SELECTED_CLASSES

# Define preprocessing transforms (same as val transforms)
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(config.IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load model globally
model = load_adapter(config, config.CHECKPOINT_DIR)
model.eval()
print("Model loaded successfully!")

def predict_fn(image: Image.Image):
    if image is None:
        return {}
    
    # Preprocess
    img = image.convert("RGB")
    tensor = val_transform(img).unsqueeze(0).to(config.DEVICE)
    
    # Predict
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1).squeeze(0)
    
    # Format for gr.Label (dict of class: confidence)
    results = {class_names[i]: float(probs[i]) for i in range(len(class_names))}
    return results

def explain_fn(image: Image.Image):
    if image is None:
        return "", None
        
    # Preprocess
    img = image.convert("RGB")
    tensor = val_transform(img).unsqueeze(0).to(config.DEVICE)
    
    # Predict
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred_idx = torch.max(probs, dim=1)
        
    conf = conf.item() * 100.0
    pred_idx = pred_idx.item()
    pred_class = class_names[pred_idx]
    
    # Generate Grad-CAM
    cam = generate_gradcam(model, tensor, pred_idx, config.DEVICE)
    
    # Save original image temporarily for overlay_gradcam
    os.makedirs(config.GRADCAM_OUTPUT_DIR, exist_ok=True)
    orig_path = os.path.join(config.GRADCAM_OUTPUT_DIR, "gradio_orig_temp.png")
    img.save(orig_path)
    
    # Save Grad-CAM overlay
    save_path = os.path.join(config.GRADCAM_OUTPUT_DIR, "gradio_temp.png")
    overlay_gradcam(orig_path, cam, save_path, pred_class, conf)
    
    prediction_text = f"{pred_class} ({conf:.2f}%)"
    return prediction_text, save_path

# Build Gradio UI
with gr.Blocks(title="LoRA Image Classifier") as demo:
    gr.Markdown("# 🌿 Plant Disease Classifier (LoRA ViT)")
    gr.Markdown("Upload a leaf image to detect diseases, or use the Explain tab to see exactly what the model focuses on!")
    
    with gr.Tabs():
        with gr.TabItem("Predict"):
            with gr.Row():
                img_input_1 = gr.Image(type="pil", label="Upload Leaf Image")
                label_output = gr.Label(num_top_classes=5, label="Predictions")
            predict_btn = gr.Button("Predict", variant="primary")
            predict_btn.click(fn=predict_fn, inputs=img_input_1, outputs=label_output)
            
        with gr.TabItem("Explain"):
            with gr.Row():
                img_input_2 = gr.Image(type="pil", label="Upload Leaf Image")
                with gr.Column():
                    explain_text = gr.Textbox(label="Prediction", interactive=False)
                    explain_output = gr.Image(type="filepath", label="Grad-CAM Overlay")
            explain_btn = gr.Button("Generate Explanation", variant="primary")
            explain_btn.click(fn=explain_fn, inputs=img_input_2, outputs=[explain_text, explain_output])

if __name__ == "__main__":
    demo.launch(share=False, server_port=7860)
