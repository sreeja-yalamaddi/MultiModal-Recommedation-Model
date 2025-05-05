# app.py

import os
import gradio as gr
import pandas as pd
from retriever import Retriever   # your Retriever class

# ─── CONFIG ────────────────────────────────────────────────────────────────
MODEL_CONFIG = {
    ("CLIP ZeroShot",   "Fashion"):    "artifacts_RAG",
    ("CLIP ZeroShot",   "Appliances"): "artifacts_zeroshot_appl",
    ("CLIP Fine tuned", "Fashion"):    "artifacts_lora",
    ("CLIP Fine tuned", "Appliances"): "artifacts_lora_appl",
}

# 2) For each category, point to its CSV filename (inside the above dir)
PRODUCT_CSV = {
    "Fashion":    "product_data.csv",
    "Appliances": "product_data_appl_full.csv",
}

TOP_K    = 5
IMG_SIZE = 150
APPROACHES = sorted({app for app, _ in MODEL_CONFIG})
CATEGORIES = sorted({cat for _, cat in MODEL_CONFIG})

# ─── 1) INSTANTIATE ONE Retriever PER (APPROACH, CATEGORY) ────────────────
retrievers = {}
for (approach, category), faiss_dir in MODEL_CONFIG.items():
    csv_path = os.path.join(faiss_dir, PRODUCT_CSV[category])
    retrievers[(approach, category)] = Retriever(
        approach    = "zero_shot" if "ZeroShot" in approach else "lora",
        faiss_dir   = faiss_dir,
        product_csv = csv_path
    )

# ─── 2) CARD RENDERER (unchanged) ─────────────────────────────────────────
def build_cards(df: pd.DataFrame, scores=None):
    cards = ""
    for idx, row in df.iterrows():
        title   = row["product_title"]
        img_url = row["product_image_url"]
        link    = row.get("product_link", "#")
        score   = scores[idx] if scores is not None else None
        score_html = (
            f"<div style='font-size:12px;color:#555;'>Score: {score:.2f}</div>"
            if score is not None else ""
        )
        cards += f"""
        <div style="display:inline-block; width:{IMG_SIZE}px; margin:8px; text-align:center;">
          <a href="{link}" target="_blank" style="text-decoration:none;color:inherit;">
            <img src="{img_url}"
                 style="width:{IMG_SIZE}px; height:{IMG_SIZE}px;
                        object-fit:cover; border:1px solid #ddd; border-radius:4px;" />
            <div style="margin-top:6px; font-size:14px; font-weight:500;">
              {title}
            </div>
            {score_html}
          </a>
        </div>
        """
    return f"<div style='display:flex; flex-wrap:wrap; justify-content:center;'>{cards}</div>"

# ─── 3) INFERENCE FUNCTION ─────────────────────────────────────────────────
def get_recommendations(text, img_url, img_file, approach, category, top_k):
    # select the correct retriever
    retriever = retrievers.get((approach, category))
    if retriever is None:
        errmsg = f"<i>No retriever for {approach} + {category}</i>"
        return errmsg, errmsg, errmsg

    # determine final image input
    final_img = None
    if img_url and img_url.strip():
        final_img = img_url.strip()
    elif img_file:
        tmp = "temp_query.jpg"
        img_file.save(tmp)
        final_img = tmp

    # 1) Text-only
    if text and text.strip():
        df_t, scores_t = retriever.query(
            input_text=text.strip(),
            input_image_path=None,
            k=top_k
        )
        html_t = build_cards(df_t, scores_t)
    else:
        html_t = "<i>No text query provided.</i>"

    # 2) Image-only
    if final_img:
        df_i, scores_i = retriever.query(
            input_text=None,
            input_image_path=final_img,
            k=top_k
        )
        html_i = build_cards(df_i, scores_i)
    else:
        html_i = "<i>No image provided.</i>"

    # 3) Text + Image
    if text and text.strip() and final_img:
        df_b, scores_b = retriever.query(
            input_text=text.strip(),
            input_image_path=final_img,
            k=top_k
        )
        html_b = build_cards(df_b, scores_b)
    else:
        html_b = "<i>Need both text and image for combined query.</i>"

    return html_t, html_i, html_b

# ─── 4) BUILD GRADIO UI ─────────────────────────────────────────────────────
with gr.Blocks(title="Multimodal Product Search") as demo:
    gr.Markdown("## 🛍️ Multimodal Recommendations")
    gr.Markdown(
        "Enter text, supply an image URL or upload, pick model & category, "
        "and click Recommend."
    )

    with gr.Row():
        with gr.Column(scale=2):
            txt_in      = gr.Textbox(label="Text Query", placeholder="e.g. “liquid eyeliner”")
            img_url_in  = gr.Textbox(label="Image URL", placeholder="https://...jpg")
            img_file_in = gr.Image(label="Upload Image", type="pil")
            approach_in = gr.Dropdown(APPROACHES, value=APPROACHES[0], label="Model Approach")
            category_in = gr.Dropdown(CATEGORIES, value=CATEGORIES[0], label="Category")
            k_in        = gr.Slider(1, 10, step=1, value=TOP_K, label="Top‑K")
            btn         = gr.Button("Recommend")

        with gr.Column(scale=3):
            gr.Markdown("### Text‑only")
            out_t = gr.HTML()
            gr.Markdown("### Image‑only")
            out_i = gr.HTML()
            gr.Markdown("### Text + Image")
            out_b = gr.HTML()

    btn.click(
        fn=get_recommendations,
        inputs=[txt_in, img_url_in, img_file_in, approach_in, category_in, k_in],
        outputs=[out_t, out_i, out_b],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7869, share=True)
