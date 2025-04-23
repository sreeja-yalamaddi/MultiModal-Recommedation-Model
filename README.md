# CLIP-Based Product Recommendation

This project implements a **multimodal product recommendation system** leveraging [OpenAI’s CLIP](https://huggingface.co/openai/clip-vit-base-patch32) model. 
It supports both **zero-shot recommendations** and **domain-adaptive fine-tuning** using LoRA (Low-Rank Adaptation), optimized with mixed precision for efficiency.

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```
## Dataset

Leveraged[Amazon reviews data 2023 from huggingface] (https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) for this study primarily using below features.

- product_title
- product_description
- product_image_url

## Features

-  **Zero-shot retrieval** using cosine similarity over CLIP embeddings.
-  **LoRA fine-tuning** to adapt CLIP to your specific product catalog.
-  **Optimized training** with mixed precision and validation monitoring.
-  **FAISS-based similarity search** for scalable and efficient retrieval.
-  **Multimodal queries** supported (text, image, or both).



## Notes

-   Embeddings are stored and indexed using FAISS.
-   Training is tested on A100 with 20k samples and ~100k samples (In progress)
-   Input queries can be text, image, or both.

