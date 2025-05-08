import os
import torch
import faiss
import pandas as pd
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from peft import get_peft_model, LoraConfig, TaskType
import requests
from io import BytesIO

class Retriever:
    def __init__(self, 
                 approach: str = "zero_shot",
                 faiss_dir: str = None,
                 product_csv: str = "product_data_appl_full.csv"):
        """
        Initializes the model, FAISS index, and product data. 
        These will be reused for queries.
        
        Parameters:
        - approach: "zero_shot", "lora", or "lora_opt"
        - faiss_dir: directory where the 'faiss.index' and model are saved
        - product_csv: path to the CSV containing product metadata
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(approach, faiss_dir)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)
        self.df = pd.read_csv(product_csv)
        self.index = self._load_faiss(faiss_dir)
    
    def _load_model(self, approach, save_dir):
        """
        Loads the model based on the specified approach ('zero_shot', 'lora', etc.)
        """
        if save_dir:
            model_path = os.path.join(save_dir, "model")
            if os.path.isdir(model_path):
                base_model = CLIPModel.from_pretrained(model_path)
                if approach == "zero_shot":
                    return base_model.to(self.device).eval()
                config = LoraConfig(
                    r=8, lora_alpha=16,
                    target_modules=["q_proj", "v_proj"],
                    lora_dropout=0.1, bias="none",
                    task_type=TaskType.FEATURE_EXTRACTION
                )
                return get_peft_model(base_model, config).to(self.device).eval()
        
        # Load fresh model
        base_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        if approach == "zero_shot":
            return base_model.to(self.device).eval()
        
        config = LoraConfig(
            r=8, lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.1, bias="none",
            task_type=TaskType.FEATURE_EXTRACTION
        )
        return get_peft_model(base_model, config).to(self.device).eval()
    
    def _load_faiss(self, save_dir):
        """
        Loads the FAISS index from the given directory.
        """
        faiss_files = [f for f in os.listdir(save_dir) if f.endswith(".index")]
        if not faiss_files:
            raise FileNotFoundError(f"No .index file found in {save_dir}")
        if len(faiss_files) > 1:
            raise ValueError(f"Multiple FAISS index files found: {faiss_files}")
        return faiss.read_index(os.path.join(save_dir, faiss_files[0]))

    def query(self, input_text=None, input_image_path=None, k=5):
        """
        Perform a query on the preloaded FAISS index and return the top results.
        
        Parameters:
        - input_text: Text input for the query
        - input_image_path: Image URL or local path
        - k: Number of results to return
        
        Returns:
        - top_df: DataFrame containing the top k results
        - top_scores: Scores for the top k results
        """
        batch = {}
        if input_text:
            tok = self.processor.tokenizer([input_text], padding=True, truncation=True, return_tensors="pt")
            batch["input_ids"] = tok.input_ids.to(self.device)
            batch["attention_mask"] = tok.attention_mask.to(self.device)
        
        if input_image_path:
            img = self._load_image(input_image_path)
            img_out = self.processor.image_processor(images=[img], return_tensors="pt")
            batch["pixel_values"] = img_out.pixel_values.to(self.device)
        
        # Forward pass to get query embedding
        with torch.no_grad():
            if "input_ids" in batch and "pixel_values" in batch:
                t_emb = self.model.get_text_features(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                i_emb = self.model.get_image_features(pixel_values=batch["pixel_values"])
                q_emb = (t_emb + i_emb) / 2
            elif "input_ids" in batch:
                q_emb = self.model.get_text_features(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            else:
                q_emb = self.model.get_image_features(pixel_values=batch["pixel_values"])

        # Normalize and search FAISS index
        # q_norm = F.normalize(q_emb, dim=-1).cpu().numpy().astype("float32")
        # scores, ids = self.index.search(q_norm, k)
        # top_df = self.df.iloc[ids[0]].reset_index(drop=True)
        # top_scores = scores[0]
        # return top_df, top_scores
        
        q_norm = F.normalize(q_emb, dim=-1).cpu().numpy().astype("float32")
        scores, ids = self.index.search(q_norm, k)      # shapes: (1, k), (1, k)
        scores = scores[0]
        ids    = ids[0]

        # 2) filter out any out‑of‑range IDs
        max_idx = len(self.df) - 1
        valid_mask = (ids >= 0) & (ids <= max_idx)
        valid_ids    = ids[valid_mask]
        valid_scores = scores[valid_mask]

        # 3) fetch only the valid rows
        top_df = self.df.iloc[valid_ids].reset_index(drop=True)

        return top_df, valid_scores

    def _load_image(self, input_image_path):
        """
        Loads an image from a URL or local path.
        """
        if input_image_path.startswith("http"):
            resp = requests.get(input_image_path, timeout=5)
            return Image.open(BytesIO(resp.content)).convert("RGB")
        else:
            return Image.open(input_image_path).convert("RGB")

def evaluate_approach_fast(
    approach_dir: str,
    ground_truth_excel: str,
    product_csv : str,
    output_excel: str = "recommendations_fast.xlsx",
    k: int = 5
):
    # 1. Load ground truth and group
    df = pd.read_excel(ground_truth_excel)
    grouped = df.groupby("Queries").agg({
        "Product_title": list,
        "Product_description": list,
        "Product_link": list,
        "Image_link": list
    }).reset_index()

    # 2. Initialize retriever
    retriever = Retriever(approach="zero_shot", faiss_dir=approach_dir, product_csv="product_data_appl_full.csv")

    # 3. Prepare output columns
    grouped["Model_rec_titles"] = None
    grouped["Model_rec_scores"] = None
    grouped["Mean_similarity"] = None

    # 4. Loop over each query
    for idx, row in grouped.iterrows():
        q = row["Queries"]
        img_url = row["Image_link"][0] if row["Image_link"] else None

        # Run the query using the pre-loaded retriever
        recs, scores = retriever.query(input_text=q, input_image_path=img_url, k=k)

        # Extract product titles and scores
        titles = recs["product_title"].tolist() if "product_title" in recs.columns else recs.iloc[:, 0].tolist()
        grouped.at[idx, "Model_rec_titles"] = titles
        grouped.at[idx, "Model_rec_scores"] = list(scores)
        grouped.at[idx, "Mean_similarity"] = float(pd.Series(scores).mean())

    # 5. Compute overall mean similarity and save
    overall_mean = float(grouped["Mean_similarity"].mean())
    grouped.to_excel(output_excel, index=False)

    return grouped, overall_mean