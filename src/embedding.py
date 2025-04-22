import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from src.config import EMBED_BATCH, DEVICE

def batch_encode_texts(texts, processor, model):
    """
    Encode a list of text strings into a single numpy array of normalized embeddings.
    """
    embs = []
    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i : i + EMBED_BATCH]
        inputs = processor(
            text=batch,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(DEVICE)
        with torch.no_grad():
            e = model.get_text_features(**inputs)
            e = F.normalize(e, dim=-1)
        embs.append(e.cpu().numpy())
    return np.vstack(embs)

def batch_encode_images(urls, processor, model):
    """
    Encode a list of image URLs into a single numpy array of normalized embeddings.
    """
    from PIL import Image
    from io import BytesIO
    import requests

    embs = []
    for i in range(0, len(urls), EMBED_BATCH):
        imgs = []
        for url in urls[i : i + EMBED_BATCH]:
            try:
                img = Image.open(BytesIO(requests.get(url, timeout=5).content))
                img = img.convert("RGB")
            except Exception:
                img = Image.new("RGB", (224, 224), "white")
            imgs.append(img)

        inputs = processor(
            images=imgs,
            return_tensors="pt",
            padding=True
        ).to(DEVICE)

        with torch.no_grad():
            e = model.get_image_features(**inputs)
            e = F.normalize(e, dim=-1)
        embs.append(e.cpu().numpy())
    return np.vstack(embs)
