# CLIP-Based Product Recommendation

This project provides a simple and extensible setup for building a **multimodal product recommendation system** using OpenAI’s CLIP model. It includes support for:

- Zero-shot retrieval via cosine similarity on CLIP embeddings.
- LoRA-based fine-tuning to adapt CLIP to product domain.
- Optimized LoRA training with mixed precision and validation loss monitoring.

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```
## Dataset

- product_title
- product_description
- product_image_url

## Usage

1. Zero-Shot Embedding + Recommendation

Generate CLIP embeddings and perform cosine similarity:

```bash
python src/main_zero_shot.py
```
2. Basic LoRA Fine-Tuning

Train LoRA layers on text-image pairs from your product data:

```bash
python src/train_lora.py
```
3. Optimized LoRA Training

Same as above but with validation monitoring and mixed precision:

```bash
python src/train_lora_opt.py
```
## Query API

See recommend.py for functions:

```
python

recommend_by_text(text_query, processor, model, df, top_k=5)
recommend_by_image(image_url, processor, model, df, top_k=5)
recommend_by_both(text, image_url, processor, model, df, alpha=0.5, top_k=5)
```
## Notes

-   Embeddings are stored and indexed using FAISS.
-   Training is tested on A100 with 200k samples.
-   Input queries can be text, image, or both.

