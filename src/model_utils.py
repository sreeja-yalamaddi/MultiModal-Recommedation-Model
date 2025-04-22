import logging
import torch
from transformers import CLIPProcessor, CLIPModel
from peft import PeftModel
from src.config import MODEL_DIR, DEVICE

def load_clip_lora():
    """
    Load CLIP + LoRA‑adapter for inference.
    Returns:
      - processor: CLIPProcessor
      - model: PeftModel‑wrapped CLIPModel on DEVICE
    """
    processor = CLIPProcessor.from_pretrained(MODEL_DIR)
    base      = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    model     = PeftModel.from_pretrained(base, MODEL_DIR).to(DEVICE)
    model.eval()
    logging.info(f"Loaded CLIP+LoRA model from {MODEL_DIR}")
    return processor, model
