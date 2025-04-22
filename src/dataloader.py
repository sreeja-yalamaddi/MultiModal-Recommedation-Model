import pandas as pd
import logging
from src.config import DATA_CSV

def load_metadata():
    """
    Load the product metadata CSV and ensure a 'product_text' column exists.
    """
    df = pd.read_csv(DATA_CSV)
    if "product_text" not in df:
        df["product_text"] = (
            df["product_title"].fillna("") + ". " + df["product_description"].fillna("")
        )
    logging.info(f"Loaded {len(df)} products from {DATA_CSV}")
    return df
