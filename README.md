# CLIP-Based Product Recommendation
  
## Project Overview

We use OpenAI’s [CLIP](https://openai.com/blog/clip) model to generate joint image-text embeddings for products. To adapt CLIP to the domain-specific nuances of product metadata, we apply **parameter-efficient fine-tuning** (LoRA) on top of frozen CLIP weights. We further explore preprocessing techniques such as summarizing verbose product metadata using **BART**, which significantly boosts retrieval performance.

> This repository shows our experiments based on:
> - Zero-shot CLIP
> - LoRA-PEFT fine-tuning (with and without mixed precision)
> - Faiss-based retrieval
> - AvgMeanSimilarity@5 evaluation metric

---

##  Dataset

We use the **Amazon Reviews 2023** dataset, Randomly selected below 2 categories for this study:
- **Fashion**
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
| Ground Truth CLIP Embedding | 0.5086 | 0.5300          |

> Evaluation metric: **AvgMeanSimilarity@5**

---

## Model Artifacts

Artifacts for each of 3 approaches for the two categories are in following paths:

1. Fashion 
> - Zero-shot - ./embeddings/artifacts_zeroshot_fashion/
> - LoRA-PEFT - ./embeddings/artifacts_lora_fashion/
> - LoRA-PEFT with MixedPrecision - ./embeddings/artifacts_lora_mp_fashion/

2. Appliances 
> - Zero-shot - ./embeddings/artifacts_zeroshot_appliances/
> - LoRA-PEFT - ./embeddings/artifacts_lora_appliances/
> - LoRA-PEFT with MixedPrecision - ./embeddings/artifacts_lora_mp_appliances/

