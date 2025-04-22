# src/train_lora.py
import torch, logging
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPModel
from peft import get_peft_model, LoraConfig, TaskType
from src.data_loader import load_metadata
from src.embedding    import batch_encode_texts, batch_encode_images  # for post‑finetune indexing

from src.config       import *
from src.model_utils  import ProductContrastiveDataset, collate_contrastive

def train():
    logging.info("Starting LoRA fine‑tuning")
    df = load_metadata()
    dataset = ProductContrastiveDataset(df)           # returns raw text+PIL
    loader  = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_contrastive)

    # 1) Load base CLIP
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    base      = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    # 2) Apply LoRA
    lora_cfg = LoraConfig(r=8, lora_alpha=16,
                          target_modules=["q_proj","v_proj"],
                          lora_dropout=0.1, bias="none",
                          task_type=TaskType.FEATURE_EXTRACTION)
    model = get_peft_model(base, lora_cfg).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    # 3) Contrastive loop
    for epoch in range(5):
        total=0
        for batch in loader:
            batch = {k:v.to(DEVICE) for k,v in batch.items()}
            text_embs  = model.get_text_features(**batch)
            image_embs = model.get_image_features(pixel_values=batch["pixel_values"])
            text_embs  = torch.nn.functional.normalize(text_embs,  dim=-1)
            image_embs = torch.nn.functional.normalize(image_embs, dim=-1)
            logits     = text_embs @ image_embs.t()
            labels     = torch.arange(len(logits), device=DEVICE)
            loss       = (torch.nn.functional.cross_entropy(logits, labels) +
                          torch.nn.functional.cross_entropy(logits.t(), labels)) / 2
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total += loss.item()
        logging.info(f"Epoch {epoch+1} loss: {total/len(loader):.4f}")

    # 4) Save adapter
    model.save_pretrained(MODEL_DIR)
    processor.save_pretrained(MODEL_DIR)

    # 5) Rebuild indices
    text_embs  = batch_encode_texts(df["product_text"], processor, model)
    image_embs = batch_encode_images(df["product_image_url"], processor, model)
    from src.index_builder import build_and_save_indices
    build_and_save_indices(text_embs, image_embs)

if __name__=="__main__":
    train()
