"""
Track A baseline system.

This version returns to our best model (bge-large-en-v1.5)
and fixes the TRUNCATION problem by using "Head-and-Tail"
encoding. This version fixes the input dictionary error.
"""
import random
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
import torch

# --- Tell pandas to use tqdm for its 'apply' function ---
tqdm.pandas()

# --- 1. Our new, smarter prediction function ---
def predict_head_and_tail(row, model, tokenizer):
    """
    Encodes the head and tail of a text, averages them,
    and then compares similarity.
    """
    anchor, text_a, text_b = row["anchor_text"], row["text_a"], row["text_b"]
    
    # --- 2. A helper function to do the head/tail encoding ---
    def get_smart_embedding(text):
        # Tokenize *without* special tokens so we can do manual head/tail
        tokens = tokenizer.encode(text, add_special_tokens=False)
        
        # Manually add the CLS and SEP tokens
        cls_token_id = tokenizer.cls_token_id
        sep_token_id = tokenizer.sep_token_id
        
        # Our target length is 512. We use 255 for head, 255 for tail.
        # This leaves 2 tokens for CLS and SEP.
        if len(tokens) > 510:
            head = tokens[:255]
            tail = tokens[-255:]
            input_tokens = [cls_token_id] + head + tail + [sep_token_id]
        else:
            input_tokens = [cls_token_id] + tokens + [sep_token_id]
            
        # Create the dictionary the model *requires*
        input_ids = torch.tensor([input_tokens]).to(model.device)
        attention_mask = torch.tensor([[1] * len(input_tokens)]).to(model.device)
        
        features = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }

        # --- 3. Get the embedding ---
        with torch.no_grad():
            # Pass the full dictionary to the model
            output_dict = model(features)
        
        # The SentenceTransformer wrapper puts the final vector here:
        return output_dict['sentence_embedding']

    # --- 4. Get the smart vectors for all three stories ---
    anchor_vec = get_smart_embedding(anchor)
    a_vec = get_smart_embedding(text_a)
    b_vec = get_smart_embedding(text_b)

    # --- 5. Calculate similarity ---
    sim_a = util.cos_sim(anchor_vec, a_vec).item()
    sim_b = util.cos_sim(anchor_vec, b_vec).item()
    
    return sim_a > sim_b


# --- CRITICAL CHANGE 1 ---
# Change this variable from "random" to "openai"
baseline = "openai"  # or "random"
# -------------------------

df = pd.read_json("data/dev_track_a.jsonl", lines=True)

if baseline == "openai":
    print("Loading Sentence Transformer model (this happens once)...")
    
    model_name = 'BAAI/bge-large-en-v1.5'
    embedder = SentenceTransformer(model_name)
    tokenizer = embedder.tokenizer
    
    print("Model loaded. Starting predictions (Head-and-Tail v2)...")

    df["predicted_text_a_is_closer"] = df.progress_apply(
        predict_head_and_tail, axis=1, args=(embedder, tokenizer)
    )
    
elif baseline == "random":
    df["predicted_text_a_is_closer"] = df.apply(
        lambda row: random.choice([True, False]), axis=1
    )

accuracy = (df["predicted_text_a_is_closer"] == df["text_a_is_closer"]).mean()
print(f"Accuracy: {accuracy:.3f}")


df["text_a_is_closer"] = df["predicted_text_a_is_closer"]
if "predicted_text_a_is_closer" in df.columns:
    del df["predicted_text_a_is_closer"]

open("output/track_a.jsonl", "w").write(df.to_json(orient='records', lines=True))
print("Successfully saved results to output/track_a.jsonl")
