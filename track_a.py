import random
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
import torch
import os

tqdm.pandas()

def predict_with_embeddings(row, model):
    anchor, text_a, text_b = row["anchor_text"], row["text_a"], row["text_b"]
    
    # Standard encode (model handles truncation automatically now)
    anchor_vec = model.encode(anchor, convert_to_tensor=True)
    a_vec = model.encode(text_a, convert_to_tensor=True)
    b_vec = model.encode(text_b, convert_to_tensor=True)

    sim_a = util.cos_sim(anchor_vec, a_vec)
    sim_b = util.cos_sim(anchor_vec, b_vec)
    
    return (sim_a > sim_b).item()

baseline = "openai" 
df = pd.read_json("data/dev_track_a.jsonl", lines=True)

if baseline == "openai":
    # Point to the folder where you will upload/unzip your model
    model_path = './my_supervised_model'
    print(f"Loading Supervised Model from '{model_path}'...")
    embedder = SentenceTransformer(model_path)
    
    print("Model loaded. Starting predictions...")
    df["predicted_text_a_is_closer"] = df.progress_apply(
        predict_with_embeddings, axis=1, args=(embedder,)
    )
    
elif baseline == "random":
    df["predicted_text_a_is_closer"] = df.apply(
        lambda row: random.choice([True, False]), axis=1
    )

accuracy = (df["predicted_text_a_is_closer"] == df["text_a_is_closer"]).mean()
print(f"Accuracy: {accuracy:.3f}")

os.makedirs("output", exist_ok=True)

df["text_a_is_closer"] = df["predicted_text_a_is_closer"]
if "predicted_text_a_is_closer" in df.columns:
    del df["predicted_text_a_is_closer"]

open("output/track_a.jsonl", "w").write(df.to_json(orient='records', lines=True))
print("Track A done.")