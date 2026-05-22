from dataclasses import dataclass, field
import torch

@dataclass
class Config:
    DATASET_ROOT: str = "data/plantvillage"
    NUM_CLASSES: int = 10
    IMAGE_SIZE: int = 224
    BATCH_SIZE: int = 32
    NUM_EPOCHS: int = 15
    LEARNING_RATE: float = 2e-4
    LORA_RANK: int = 8
    LORA_ALPHA: int = 16
    LORA_DROPOUT: float = 0.1
    LORA_TARGET_MODULES: list = field(default_factory=lambda: ["query", "value"])
    TRAIN_SPLIT: float = 0.7
    VAL_SPLIT: float = 0.15
    TEST_SPLIT: float = 0.15
    CHECKPOINT_DIR: str = "outputs/checkpoints"
    GRADCAM_OUTPUT_DIR: str = "outputs/gradcam_samples"
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    RANDOM_SEED: int = 42
