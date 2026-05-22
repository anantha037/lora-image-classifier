import os
import sys
import torch
import torch.nn as nn
import timm
from peft import LoraConfig, get_peft_model, PeftModel

# Append parent dir for importing Config if necessary
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

def print_layer_names(model: nn.Module, keywords: list = ["query", "value", "qkv"]):
    """
    Inspects and prints all layer names in the model that contain any of the given keywords.
    Helps verify if the target modules for LoRA actually exist in the model's architecture.
    """
    print("--- Inspecting Model Layer Names ---")
    found = False
    for name, module in model.named_modules():
        if any(keyword in name for keyword in keywords):
            print(f"Found matching layer: {name} ({type(module).__name__})")
            found = True
    if not found:
        print(f"No layers found matching keywords: {keywords}")
    print("------------------------------------")

def load_base_model(config: Config) -> nn.Module:
    """
    Loads the timm ViT base model, replaces the classification head, 
    and completely freezes the base model weights.
    """
    # Load pretrained ViT
    model = timm.create_model("vit_base_patch16_224", pretrained=True)
    
    # Freeze all weights
    for param in model.parameters():
        param.requires_grad = False
        
    # Replace the classification head (this unfreezes the head by default)
    # The head in timm's ViT is named 'head'
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, config.NUM_CLASSES)
    
    # Calculate parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Base Model Loaded - Total Parameters: {total_params:,} | Trainable: {trainable_params:,}")
    return model

def get_model(config: Config) -> PeftModel:
    """
    Full pipeline to load the ViT model, freeze it, apply PEFT LoRA, 
    and move to the configured device.
    """
    base_model = load_base_model(config)
    
    # Inspect layers to see what matches our target modules
    # timm ViT usually has "qkv" layers instead of separate "query" and "value"
    print_layer_names(base_model, keywords=["query", "value", "qkv"])
    
    # Setup LoRA Config
    lora_config = LoraConfig(
        r=config.LORA_RANK,
        lora_alpha=config.LORA_ALPHA,
        target_modules=config.LORA_TARGET_MODULES,
        lora_dropout=config.LORA_DROPOUT,
        bias="none",
        modules_to_save=["head"],
    )
    
    # Apply PEFT
    peft_model = get_peft_model(base_model, lora_config)
    
    # Move to device
    peft_model = peft_model.to(config.DEVICE)
    
    return peft_model

def save_adapter(model: PeftModel, path: str):
    """
    Saves ONLY the LoRA adapter weights, NOT the full base model.
    """
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    
    # Check file size of adapter files
    size_mb = 0
    for f in os.listdir(path):
        filepath = os.path.join(path, f)
        if os.path.isfile(filepath):
            size_mb += os.path.getsize(filepath) / (1024 * 1024)
            
    print(f"Adapter successfully saved to {path} (Size: {size_mb:.2f} MB)")

def load_adapter(config: Config, adapter_path: str) -> PeftModel:
    """
    Reconstructs the base model and loads the saved PEFT adapter.
    Returns the model in eval mode on the correct device.
    """
    # 1. Reconstruct base model
    base_model = load_base_model(config)
    
    # 2. Load PEFT adapter
    peft_model = PeftModel.from_pretrained(base_model, adapter_path)
    peft_model = peft_model.to(config.DEVICE)
    peft_model.eval()
    
    return peft_model

def get_model_summary(model: PeftModel):
    """
    Prints total parameters, trainable parameters, trainable percentage, 
    and lists specific layer names that have LoRA applied.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_pct = (trainable_params / total_params) * 100
    
    print("\n--- Model Summary ---")
    print(f"Total Parameters:     {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Trainable Percentage: {trainable_pct:.4f}%")
    model.print_trainable_parameters()
    
    print("\nLayers with LoRA applied:")
    lora_layers = [name for name, _ in model.named_modules() if "lora" in name.lower()]
    if not lora_layers:
        print("  WARNING: No LoRA layers found! Check config.LORA_TARGET_MODULES against actual layer names.")
    else:
        # Show a summary of adapter layers, avoiding printing every single nested module
        printed = set()
        for name in lora_layers:
            if "lora_A" in name or "lora_B" in name:
                base_name = name.split(".lora_")[0]
                if base_name not in printed:
                    print(f"  - {base_name}")
                    printed.add(base_name)
    print("---------------------\n")

if __name__ == "__main__":
    print("Running Model Sanity Check...\n")
    config = Config()
    
    # Fix target modules if they default to query/value which don't exist in timm's ViT
    if config.LORA_TARGET_MODULES == ["query", "value"]:
        print("Note: Default Config target_modules are ['query', 'value']. Timm ViT uses 'qkv'.")
        print("Forcing config.LORA_TARGET_MODULES = ['qkv'] for this sanity check.")
        config.LORA_TARGET_MODULES = ["qkv"]

    model = get_model(config)
    get_model_summary(model)
    
    print(f"Model loaded and moved to {config.DEVICE}")
    print("Performing dummy forward pass...")
    
    dummy_input = torch.randn(2, 3, 224, 224).to(config.DEVICE)
    
    with torch.no_grad():
        output = model(dummy_input)
        
    print(f"Dummy Output Shape: {output.shape} (Expected: torch.Size([2, {config.NUM_CLASSES}]))")
    
    if output.shape == (2, config.NUM_CLASSES):
        print("Sanity check passed! Model is ready.")
    else:
        print("Sanity check failed! Unexpected output shape.")
