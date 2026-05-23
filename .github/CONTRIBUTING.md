# Contributing to LoRA Plant Disease Classifier

## Setup
1. Fork the repository
2. Clone your fork: `git clone https://github.com/<your-username>/lora-image-classifier`
3. Install dependencies: `pip install -r requirements.txt`

## Adding New Plant Disease Classes
- Update `SELECTED_CLASSES` in `config.py`
- Update `NUM_CLASSES` accordingly
- Retrain using `notebooks/colab_training.ipynb`

## Running Tests
- Grad-CAM demo: `python src/gradcam.py`
- API: `uvicorn api.main:app --reload --port 8080`
- Gradio demo: `python app.py`

## Pull Request Guidelines
- One feature per PR
- Add docstrings to all new functions
- Test locally before submitting
