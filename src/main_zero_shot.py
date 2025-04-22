# src/main_zero_shot.py
import logging
from src.config       import *
from src.data_loader  import load_metadata
from src.model_utils  import load_pretrained_clip  # returns processor, model
from src.embedding    import batch_encode_texts, batch_encode_images
from src.index_builder import build_and_save_indices
from src.recommend    import recommend_by_text, recommend_by_image, recommend_by_both

def main():
    logging.info("Zero‑Shot pipeline start")
    df, = load_metadata()
    processor, model = load_pretrained_clip()  # no LoRA
    # 1) Embed catalog  
    text_embs  = batch_encode_texts(df["product_text"], processor, model)
    image_embs = batch_encode_images(df["product_image_url"], processor, model)
    # 2) Build FAISS  
    build_and_save_indices(text_embs, image_embs)
    # 3) Demo  
    print(recommend_by_text("hydrating cream", processor, model, df))
    print(recommend_by_image(df.iloc[0]["product_image_url"], processor, model, df))
    print(recommend_by_both("anti‑aging serum", df.iloc[0]["product_image_url"], processor, model, df))

if __name__ == "__main__":
    main()
