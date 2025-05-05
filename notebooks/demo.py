import gradio as gr
import torch
import os
from utils import unified_query  # Assuming your model inference function
import pandas as pd

# Modify this to your actual model folders
MODEL_DIRS = {
    "Baseline":      "artifacts_zero_shot",
    "Fine-Tuned":    "artifacts_lora",
    "Fine-Tuned MP": "/artifacts_20k_clip_mp"
}

TOP_K = 5  # Number of recommendations to show

def build_cards(df, scores=None):
    cards = ""
    for idx, row in df.iterrows():
        img_url = row["product_image_url"]
        title = row["product_title"]
        link = row.get("product_link", "#")
        score_text = f"<div style='font-size:12px; color:#555;'>Score: {scores[idx]:.2f}</div>" if scores is not None else ""

        cards += f"""
        <div style="display:inline-block; width:180px; margin:10px; text-align:center;">
            <a href="{link}" target="_blank" style="text-decoration:none; color:inherit;">
                <img src="{img_url}" alt="img" style="width:100%; height:auto; border:1px solid #ccc; border-radius:6px;">
                <div style="margin-top:6px; font-size:14px; font-weight:500;">{title}</div>
                {score_text}
            </a>
        </div>
        """
    return f"<div style='display:flex; flex-wrap:wrap; justify-content:center;'>{cards}</div>"

def get_recommendations(query_text, query_image, selected_models):
    results = {}
    
    # Save image temporarily if present
    image_path = None
    if query_image is not None:
        image_path = "temp_input.jpg"
        query_image.save(image_path)

    for model_name in selected_models:
        save_dir = MODEL_DIRS[model_name]
        recs, scores = unified_query(input_text=query_text, input_image_path=image_path, save_dir=save_dir, k=TOP_K)
        results[model_name] = build_cards(recs, scores)

    return tuple(results.get(m, "") for m in MODEL_DIRS if m in selected_models)

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("## Product Recommendation Demo (Text + Image)")
    
    with gr.Row():
        with gr.Column():
            query_text = gr.Textbox(label="Enter product query", placeholder="e.g. 'wireless earbuds for gym'")
            query_image = gr.Image(type="pil", label="Upload product image (optional)")
            selected_models = gr.CheckboxGroup(
                choices=list(MODEL_DIRS.keys()),
                label="Select models to run",
                value=list(MODEL_DIRS.keys())  # default all
            )
            run_button = gr.Button("Get Recommendations")

        with gr.Column():
            output_1 = gr.HTML()
            output_2 = gr.HTML()
            output_3 = gr.HTML()

    run_button.click(fn=get_recommendations, 
                     inputs=[query_text, query_image, selected_models], 
                     outputs=[output_1, output_2, output_3])

demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
