# CLIP-Based Product Recommendation

This project implements a **multimodal product recommendation system** leveraging [OpenAI’s CLIP](https://huggingface.co/openai/clip-vit-base-patch32) model. 
It supports both **zero-shot recommendations** and **domain-adaptive fine-tuning** using LoRA (Low-Rank Adaptation), optimized with mixed precision for efficiency.

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```
## Dataset

Leveraged [Amazon reviews data 2023 from huggingface](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) for this study primarily using below features.

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
  
## Project Overview

We use OpenAI’s [CLIP](https://openai.com/blog/clip) model to generate joint image-text embeddings for products. To adapt CLIP to the domain-specific nuances of product metadata, we apply **parameter-efficient fine-tuning** (LoRA) on top of frozen CLIP weights. We further explore preprocessing techniques such as summarizing verbose product metadata using **BART**, which significantly boosts retrieval performance.

> This repository supports experiments with:
> - Zero-shot CLIP
> - LoRA-PEFT fine-tuning (with and without mixed precision)
> - Faiss-based retrieval
> - AvgMeanSimilarity@5 evaluation metric

---

##  Dataset

We use the **Amazon Reviews 2023** dataset, Randomly selected below 2 categories for this study:
- **Fashion **
- **Home Appliances**

Each product consists of:
- Image(s)
- Metadata: title, brand, description, etc.

### Data Preprocessing
- Used `facebook/bart-large-cnn` to summarize metadata into concise product descriptions.
- Cleaned and tokenized summaries.
- Resized images to 224×224 (CLIP-compatible).
- Retained one representative image per product.

## Training setup:
-	Batch size: 128
-	Learning rate: 2e-5
-	Epochs: 10
- Optimizer: AdamW
-	Hardware: NVIDIA A100 (40 GB)
  
---

##  Model Variants

| Model Type          | Fashion Score | Appliances Score |
|---------------------|---------------|------------------|
| CLIP Zero-Shot      | 0.6725        | 0.8498           |
| LoRA Fine-Tuned     | 0.7939        | 0.9073           |
| LoRA + MixedPrecision | 0.7255     | 0.8746           |
| Ground Truth CLIP Embedding | 0.5000 | 0.5300          |

> Evaluation metric: **AvgMeanSimilarity@5**

---
