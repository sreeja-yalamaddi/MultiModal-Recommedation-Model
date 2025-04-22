import os
import faiss
import torch
import torch.nn.functional as F
from src.config import INDEX_DIR, DEVICE, TOP_K, ALPHA

# Load FAISS indices
_text_index  = faiss.read_index(os.path.join(INDEX_DIR, "text.idx"))
_image_index = faiss.read_index(os.path.join(INDEX_DIR, "image.idx"))

def recommend_by_text(query, processor, model, df, top_k=TOP_K):
    inputs = processor(
        text=[query],
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(DEVICE)
    with torch.no_grad():
        emb = model.get_text_features(**inputs)
        emb = F.normalize(emb, dim=-1).cpu().numpy()
    D, I = _image_index.search(emb, top_k)
    return df.iloc[I[0]][["product_title", "product_image_url"]].assign(score=D[0])

def recommend_by_image(url, processor, model, df, top_k=TOP_K):
    from PIL import Image
    from io import BytesIO
    import requests

    img = Image.open(BytesIO(requests.get(url, timeout=5).content)).convert("RGB")
    inputs = processor(
        images=[img],
        return_tensors="pt",
        padding=True
    ).to(DEVICE)
    with torch.no_grad():
        emb = model.get_image_features(**inputs)
        emb = F.normalize(emb, dim=-1).cpu().numpy()
    D, I = _text_index.search(emb, top_k)
    return df.iloc[I[0]][["product_title", "product_image_url"]].assign(score=D[0])

def recommend_by_both(query, url, processor, model, df, alpha=ALPHA, top_k=TOP_K):
    # Text embed
    t_in = processor(
        text=[query],
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(DEVICE)
    # Image embed
    from PIL import Image
    from io import BytesIO
    import requests
    img = Image.open(BytesIO(requests.get(url, timeout=5).content)).convert("RGB")
    i_in = processor(images=[img], return_tensors="pt", padding=True).to(DEVICE)

    with torch.no_grad():
        t_emb = model.get_text_features(**t_in); t_emb = F.normalize(t_emb, dim=-1).cpu().numpy()
        i_emb = model.get_image_features(**i_in); i_emb = F.normalize(i_emb, dim=-1).cpu().numpy()

    D_i, I_i = _image_index.search(i_emb, top_k)
    D_t, I_t = _image_index.search(t_emb, top_k)

    merged = {}
    for idx, sc in zip(I_i[0], D_i[0]):
        merged[idx] = merged.get(idx, 0.0) + alpha * sc
    for idx, sc in zip(I_t[0], D_t[0]):
        merged[idx] = merged.get(idx, 0.0) + (1 - alpha) * sc

    top = sorted(merged.items(), key=lambda x: x[1], reverse=True)[:top_k]
    idxs, scores = zip(*top)
    res = df.iloc[list(idxs)][["product_title", "product_image_url"]].copy()
    res["score"] = scores
    return res.reset_index(drop=True)
