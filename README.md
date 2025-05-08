# CLIP-Based Multimodal Product Recommendation

##  Project Overview

We use OpenAI’s [CLIP](https://openai.com/blog/clip) model to generate joint image-text embeddings for products. To adapt CLIP to domain-specific nuances in e-commerce, we apply **parameter-efficient fine-tuning** (LoRA) on top of frozen CLIP weights. We also explore text preprocessing via **BART-based summarization**, which significantly boosts retrieval performance.

> This repository demonstrates:
> - Zero-shot CLIP retrieval
> - LoRA-PEFT fine-tuning (standard and mixed-precision)
> - FAISS-based similarity search
> - Custom evaluation using **AvgMeanSimilarity@5**

---

##  Dataset

We use the **Amazon Reviews 2023** dataset, selecting two categories:

- `Beauty`
- `Home Appliances`

Each product contains:
- At least one image
- Metadata: title, brand, description, etc.

###  Data Preprocessing

- Summarized product metadata using `facebook/bart-large-cnn`
- Cleaned and tokenized the summaries
- Resized images to **224×224** (CLIP input format)
- Retained one representative image per product

> Cleaned data files:
> - `./input_data/product_data_beauty.csv`
> - `./input_data/product_data_appliances.csv`

---

##  Training Setup

- Batch size: 128  
- Learning rate: 2e-5  
- Epochs: 10  
- Optimizer: AdamW  
- Hardware: NVIDIA A100 (40 GB)

---

##  Model Variants & Evaluation

| Model Type              | Beauty Score | Appliances Score |
|-------------------------|--------------|------------------|
| CLIP Zero-Shot          | 0.6725       | 0.8498           |
| LoRA Fine-Tuned         | 0.7939       | 0.9073           |
| LoRA + Mixed Precision  | 0.7255       | 0.8746           |
| Ground Truth Embedding  | 0.5086       | 0.5300           |

> Metric: **AvgMeanSimilarity@5**

---

##  Notebooks & Scripts

| File | Description |
|------|-------------|
| `scripts/embedding_based_approach.ipynb` | Preprocessing, zero-shot CLIP embeddings, FAISS indexing |
| `scripts/Finetune CLIP using LoRA.ipynb` | LoRA-PEFT training and embedding generation |
| `scripts/Finetune CLIP LORA MP.ipynb`    | LoRA + Mixed Precision training |
| `scripts/Model Eval.ipynb`               | Inference & evaluation for both categories |
| `scripts/ground_truth_similarity.ipynb`  | Evaluation using handcrafted ground-truth queries |
| `scripts/evaluate.py`  | core retrieval engine used by both the demo app and for batch evaluation |
| `scripts/app.py` | Gradio-based UI for interactive multimodal product search |

> Ground truth queries: `./ground_truth/`

---

##  Model Artifacts

| Category   | Model Variant           | Path |
|------------|-------------------------|------|
| Beauty     | Zero-shot               | `./embeddings/artifacts_zeroshot_beauty/` |
| Beauty     | LoRA-PEFT               | `./embeddings/artifacts_lora_beauty/` |
| Beauty     | LoRA + Mixed Precision  | `./embeddings/artifacts_lora_mp_beauty/` |
| Appliances | Zero-shot               | `./embeddings/artifacts_zeroshot_appliances/` |
| Appliances | LoRA-PEFT               | `./embeddings/artifacts_lora_appliances/` |
| Appliances | LoRA + Mixed Precision  | `./embeddings/artifacts_lora_mp_appliances/` |

---

##  Gradio Demo

Try out the multimodal recommendation demo here:  
 [Gradio Space Link](https://huggingface.co/spaces/Sreeja05/MultiModalRecommendations)

> Input a product description, image URL or upload an image, choose a model variant and category, and see top-K matching products.



