# src/train_lora_opt.py
import os, torch, logging
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
from peft import get_peft_model, LoraConfig, TaskType
from transformers import CLIPProcessor, CLIPModel
from src.data_loader import load_metadata, ProductContrastiveDataset, collate_contrastive
from src.config      import *

def train_opt():
    logging.info("Optimized LoRA training start")
    df = load_metadata()
    ds = ProductContrastiveDataset(df)
    n_val = int(0.1*len(ds)); n_tr = len(ds)-n_val
    train_ds, val_ds = random_split(ds, [n_tr, n_val], generator=torch.Generator().manual_seed(42))
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_contrastive, num_workers=4, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=32, shuffle=False,collate_fn=collate_contrastive, num_workers=4, pin_memory=True)

    # Model & LoRA
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    base      = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    lora_cfg  = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj","v_proj"], lora_dropout=0.1, bias="none", task_type=TaskType.FEATURE_EXTRACTION)
    model     = get_peft_model(base, lora_cfg).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    scaler    = GradScaler()

    best_val = float("inf")
    for epoch in range(1, 11):
        # Train
        model.train(); train_loss=0
        for batch in train_dl:
            batch = {k:v.to(DEVICE) for k,v in batch.items()}
            optimizer.zero_grad()
            with autocast("cuda"):
                t_emb = model.get_text_features(**batch)
                i_emb = model.get_image_features(pixel_values=batch["pixel_values"])
                t_emb = torch.nn.functional.normalize(t_emb, dim=-1)
                i_emb = torch.nn.functional.normalize(i_emb, dim=-1)
                logits = t_emb @ i_emb.t()
                labels = torch.arange(len(logits), device=DEVICE)
                loss   = (torch.nn.functional.cross_entropy(logits, labels) +
                          torch.nn.functional.cross_entropy(logits.t(), labels)) / 2
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
            train_loss += loss.item()

        # Validate
        model.eval(); val_loss=0
        with torch.no_grad(), autocast("cuda"):
            for batch in val_dl:
                batch = {k:v.to(DEVICE) for k,v in batch.items()}
                t_emb = model.get_text_features(**batch)
                i_emb = model.get_image_features(pixel_values=batch["pixel_values"])
                t_emb = torch.nn.functional.normalize(t_emb, dim=-1)
                i_emb = torch.nn.functional.normalize(i_emb, dim=-1)
                logits = t_emb @ i_emb.t()
                labels = torch.arange(len(logits), device=DEVICE)
                loss   = (torch.nn.functional.cross_entropy(logits, labels) +
                          torch.nn.functional.cross_entropy(logits.t(), labels)) / 2
                val_loss += loss.item()

        avg_train = train_loss/len(train_dl)
        avg_val   = val_loss/len(val_dl)
        logging.info(f"Epoch {epoch}: train={avg_train:.4f}, val={avg_val:.4f}")

        # Early stopping + checkpoint
        ckpt_dir = f"checkpoints/epoch_{epoch}"
        os.makedirs(ckpt_dir, exist_ok=True)
        model.save_pretrained(ckpt_dir); processor.save_pretrained(ckpt_dir)

        if avg_val < best_val:
            best_val = avg_val
        else:
            logging.info("Validation no improvement — early stopping")
            break

    # After training, re-index
    from src.embedding    import batch_encode_texts, batch_encode_images
    from src.index_builder import build_and_save_indices
    text_embs  = batch_encode_texts(df["product_text"], processor, model)
    image_embs = batch_encode_images(df["product_image_url"], processor, model)
    build_and_save_indices(text_embs, image_embs)

if __name__=="__main__":
    train_opt()
