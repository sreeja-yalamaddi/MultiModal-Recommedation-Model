import os

# Paths
BASE_DIR     = os.path.dirname(os.path.dirname(__file__))
DATA_CSV     = os.path.join(BASE_DIR, "data", "beauty_metadata.csv")
MODEL_DIR    = os.path.join(BASE_DIR, "clip-lora-beauty")
INDEX_DIR    = os.path.join(BASE_DIR, "indexes")

# Device
DEVICE       = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"

# Embedding & Retrieval
EMBED_BATCH  = 128
TOP_K        = 5
ALPHA        = 0.5  # for hybrid recommendations
