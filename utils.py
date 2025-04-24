# src/utils.py

import os
import pandas as pd
import torch
import torch.nn.functional as F
import faiss
import requests
from io import BytesIO
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPModel
from peft import get_peft_model, LoraConfig, TaskType

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
MODEL_NAME = "openai/clip-vit-base-patch32"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------------
# 1) DATA LOADING & CLEANING
# -----------------------------------------------------------------------------
DEFAULT_KEYWORDS = [
    'nail', 'shampoo', 'conditioner', 'eye', 'lip', 'ear', 'nose',
    'beauty', 'cosmetic', 'hair', 'skin', 'hand', 'leg', 'oil',
    'makeup', 'lotion', 'cream', 'cleanser', 'moisturizer'
]

def contains_keywords(text: str, keywords=DEFAULT_KEYWORDS) -> bool:
    """Check if any keyword appears in the text."""
    return any(k.lower() in text.lower() for k in keywords)

def create_product_text(row: pd.Series) -> str:
    """Combine title & description into a single product_text string."""
    title = str(row.get("product_title", "")).strip()
    desc  = str(row.get("product_description", "")).strip()
    if desc:
        txt = f"Product title is: {title}\nProduct description is: {desc}"
    else:
        txt = f"Product title is: {title}"
    return txt[:512]  # truncate to 512 chars

def load_and_clean_data(csv_path,keywords=DEFAULT_KEYWORDS) -> pd.DataFrame:
    """
    Load CSV, filter by keywords in title/description,
    extract product_text, and ensure valid image URLs.
    """
    df = pd.read_csv(csv_path)
    df["product_title"]       = df["product_title"].fillna("")
    df["product_description"] = df["product_description"].fillna("")

    # filter by keywords
    mask = df["product_title"].apply(lambda x: contains_keywords(x, keywords)) | \
           df["product_description"].apply(lambda x: contains_keywords(x, keywords))
    df = df[mask].copy()

    # build product_text
    df["product_text"] = df.apply(create_product_text, axis=1)

    # extract first valid image URL from product_images dict if present
    # def extract_first_valid(img_dict):
    #     invalid = {'', 'none', 'n/a', 'na'}
    #     if not isinstance(img_dict, dict):
    #         return None
    #     for key in ("hi_res","large","thumb"):
    #         for url in img_dict.get(key, []):
    #             if isinstance(url, str) and url.strip().lower() not in invalid:
    #                 return url.strip()
    #     return None

    # if "product_images" in df.columns:
    #     df["product_image_url"] = df["product_images"].apply(extract_first_valid)
    # keep only rows with text & image URL
    df = df[df["product_text"].str.strip().astype(bool) & df["product_image_url"].str.strip().astype(bool)]
    return df.reset_index(drop=True)


# -----------------------------------------------------------------------------'
0
# 2) MODEL LOADING & SAVING
# -----------------------------------------------------------------------------
def get_model(
    approach: str = "zero_shot",
    save_dir: str = None
) -> torch.nn.Module:
    """
    Load CLIP model.
      - approach: "zero_shot", "lora", or "lora_opt"
      - if save_dir is provided and 'model/' exists there, loads instead of fresh.
    """
    # try loading from disk
    if save_dir:
        mpath = os.path.join(save_dir, "model")
        if os.path.isdir(mpath):
            base = CLIPModel.from_pretrained(mpath)
            if approach == "zero_shot":
                base.to(DEVICE).eval()
                return base
            # else fine-tuned LoRA
            cfg = LoraConfig(
                r=8, lora_alpha=16,
                target_modules=["q_proj","v_proj"],
                lora_dropout=0.1, bias="none",
                task_type=TaskType.FEATURE_EXTRACTION
            )
            model = get_peft_model(base, cfg)
            model.to(DEVICE).eval()
            return model

    # fresh load
    base = CLIPModel.from_pretrained(MODEL_NAME)
    if approach == "zero_shot":
        base.to(DEVICE).eval()
        return base

    cfg = LoraConfig(
        r=8, lora_alpha=16,
        target_modules=["q_proj","v_proj"],
        lora_dropout=0.1, bias="none",
        task_type=TaskType.FEATURE_EXTRACTION
    )
    model = get_peft_model(base, cfg)
    model.to(DEVICE).eval()
    return model

def save_model_and_processor(model, save_dir: str):
    """Save fine-tuned model + processor under save_dir/model/"""
    os.makedirs(save_dir, exist_ok=True)
    out = os.path.join(save_dir, "model")
    model.save_pretrained(out)
    CLIPProcessor.from_pretrained(MODEL_NAME).save_pretrained(out)


# -----------------------------------------------------------------------------
# 3) EMBEDDINGS I/O
# -----------------------------------------------------------------------------
def save_embeddings(text_embs: torch.Tensor, image_embs: torch.Tensor, save_dir: str):
    """Save tensors to disk for later reuse."""
    os.makedirs(save_dir, exist_ok=True)
    torch.save(text_embs,  os.path.join(save_dir, "text_embs.pt"))
    torch.save(image_embs, os.path.join(save_dir, "image_embs.pt"))

def load_embeddings(save_dir: str):
    """Load saved embeddings back onto DEVICE."""
    te = torch.load(os.path.join(save_dir, "text_embs.pt")).to(DEVICE)
    ie = torch.load(os.path.join(save_dir, "image_embs.pt")).to(DEVICE)
    return te, ie


# -----------------------------------------------------------------------------
# 4) FAISS INDEX BUILD / LOAD
# -----------------------------------------------------------------------------
def build_faiss_index(embeddings: torch.Tensor, save_dir: str) -> str:
    """
    Build an IndexFlatIP over L2-normalized embeddings.
    Saves to save_dir/faiss.index and returns that path.
    """
    os.makedirs(save_dir, exist_ok=True)
    idx   = faiss.IndexFlatIP(embeddings.shape[1])
    idx.add(embeddings)
    path = os.path.join(save_dir, "faiss.index")
    faiss.write_index(idx, path)
    return path

def load_faiss_index(save_dir: str):
    """Load FAISS index from save_dir/faiss.index"""
    index_files = [f for f in os.listdir(save_dir) if f.endswith(".index")]
    if not index_files:
        raise FileNotFoundError(f"No .index file found in {save_dir}")
    if len(index_files) > 1:
        raise ValueError(f"Multiple .index files found: {index_files}. Please keep only one.")

    index_path = os.path.join(save_dir, index_files[0])
    return faiss.read_index(index_path)

# -----------------------------------------------------------------------------
# 5) EMBEDDING GENERATION
# -----------------------------------------------------------------------------
def generate_embeddings(
    model: torch.nn.Module,
    dataset,
    batch_size: int = 128,
    num_workers: int = 4
):
    """
    Run model.get_text_features and model.get_image_features over dataset,
    return (text_embs, image_embs) as [N,D] torch tensors.
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=dataset.collate_fn
    )

    t_list, i_list = [], []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Generating embeddings"):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            t = model.get_text_features(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )
            i = model.get_image_features(pixel_values=batch["pixel_values"])
            t_list.append(F.normalize(t, p=2, dim=-1).cpu())
            i_list.append(F.normalize(i, p=2, dim=-1).cpu())
    return torch.cat(t_list, dim=0), torch.cat(i_list, dim=0)


# -----------------------------------------------------------------------------
# 6) UNIFIED MULTIMODAL QUERY
# -----------------------------------------------------------------------------
def unified_query(
    input_text: str = None,
    input_image_path: str = None,
    approach: str = "zero_shot",
    save_dir: str = None,
    k: int = 5
):
    """
    Query the FAISS index for a given approach (zero_shot, lora, or lora_opt).
    """
    
    model = get_model(approach=approach, save_dir=save_dir)
    model.to(DEVICE).eval()

    df  = pd.read_csv("product_data.csv")
    idx = load_faiss_index(save_dir)

    # 2) Load processor, then split tokenizer and image_processor
    proc = CLIPProcessor.from_pretrained(MODEL_NAME, use_fast=True)
    tokenizer       = proc.tokenizer
    image_processor = proc.image_processor

    # 3) Prepare inputs
    # We batch as a single‐element batch so outputs remain [1, D]
    batch = {}
    if input_text:
        tok = tokenizer(
            [input_text],
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        batch["input_ids"]      = tok.input_ids.to(DEVICE)
        batch["attention_mask"] = tok.attention_mask.to(DEVICE)

    if input_image_path:
        if input_image_path.startswith("http"):
            resp = requests.get(input_image_path, timeout=5)
            img  = Image.open(BytesIO(resp.content)).convert("RGB")
        else:
            img  = Image.open(input_image_path).convert("RGB")

        img_out = image_processor(
            images=[img],
            return_tensors="pt"
        )
        batch["pixel_values"] = img_out.pixel_values.to(DEVICE)

    # 4) Forward pass
    with torch.no_grad():
        if "input_ids" in batch and "pixel_values" in batch:
            t_emb = model.get_text_features(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )
            i_emb = model.get_image_features(
                pixel_values=batch["pixel_values"]
            )
            q_emb = (t_emb + i_emb) / 2
        elif "input_ids" in batch:
            q_emb = model.get_text_features(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )
        else:
            q_emb = model.get_image_features(
                pixel_values=batch["pixel_values"]
            )

    # 5) Normalize, search FAISS
    q_norm = F.normalize(q_emb, dim=-1).cpu().numpy().astype("float32")
    scores, ids = idx.search(q_norm, k)

    top_df     = df.iloc[ids[0]].reset_index(drop=True)
    top_scores = scores[0]
    return top_df, top_scores


# -----------------------------------------------------------------------------
# 7) BATCH EVALUATION ON AMAZON QUERIES
# -----------------------------------------------------------------------------
def evaluate_model_on_queries(
    query_file: str = "Amazon_recom_queries.xlsx",
    save_dir: str = "artifacts_zero_shot",
    k: int = 5
) -> pd.DataFrame:
    """
    Load ground-truth from query_file, run unified_query for each unique query,
    and return a DataFrame with Amazon GT + model recs + scores.
    """
    df_q = pd.read_excel(query_file)
    grouped = df_q.groupby("Queries").agg({
        "Product_title":       list,
        "Product_description": list,
        "Product_link":        list,
        "Image_link":          list
    }).reset_index()

    for col in ["Model_rec_titles","Model_rec_descriptions","Model_rec_links","Model_rec_scores"]:
        grouped[col] = None

    for i, row in tqdm(grouped.iterrows(), total=len(grouped), desc="Eval queries"):
        q      = row["Queries"]
        img_url= row["Image_link"][0] if row["Image_link"] else None
        try:
            recs, scores = unified_query(q, img_url, save_dir, k)
            grouped.at[i, "Model_rec_titles"]       = recs["product_title"].tolist()
            grouped.at[i, "Model_rec_descriptions"] = recs["product_description"].tolist()
            grouped.at[i, "Model_rec_links"]        = recs.get("product_link", pd.Series()).tolist()
            grouped.at[i, "Model_rec_scores"]       = scores.tolist()
        except Exception as e:
            print(f"Query `{q}` failed: {e}")
            continue

    return grouped
