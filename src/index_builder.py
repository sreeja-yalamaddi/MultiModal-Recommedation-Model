import os
import faiss
import logging
from src.config import INDEX_DIR

def build_and_save_indices(text_embs, image_embs):
    """
    Build FAISS Inner-Product indices for text & image embeddings and save them.
    """
    d = text_embs.shape[1]
    txt_idx = faiss.IndexFlatIP(d)
    img_idx = faiss.IndexFlatIP(d)

    txt_idx.add(text_embs)
    img_idx.add(image_embs)

    os.makedirs(INDEX_DIR, exist_ok=True)
    faiss.write_index(txt_idx, os.path.join(INDEX_DIR, "text.idx"))
    faiss.write_index(img_idx, os.path.join(INDEX_DIR, "image.idx"))

    logging.info(f"FAISS indices saved to {INDEX_DIR}")
