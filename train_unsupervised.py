"""
Script 1: Unsupervised Fine-Tuning (Domain Adaptation)

This script takes our champion 'bge-large' model and
re-trains it (unsupervised) on the "library" of texts
from dev_track_b.jsonl.

This will be VERY SLOW, but it only needs to be run once.
"""
import pandas as pd
from sentence_transformers import SentenceTransformer, models, losses
from sentence_transformers.readers import InputExample
from torch.utils.data import DataLoader

# --- 1. Load the "Library" of Texts ---
print("Loading dev_track_b.jsonl...")
df = pd.read_json("data/dev_track_b.jsonl", lines=True)
texts = df["text"].dropna().unique().tolist()
print(f"Found {len(texts)} unique texts for training.")

# --- 2. Create Training Examples ---
train_examples = [InputExample(texts=[s, s]) for s in texts]

# --- 3. Load the Base Model ---
print("Loading base model 'BAAI/bge-large-en-v1.5'...")
base_model_name = 'BAAI/bge-large-en-v1.5'
word_embedding_model = models.Transformer(base_model_name)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# --- 4. Set up the TSDAE Dataloader and Loss ---
# This dataloader automatically "damages" the text
train_dataloader = DataLoader(train_examples, batch_size=8, shuffle=True)
train_loss = losses.DenoisingAutoEncoderLoss(
    model,
    decoder_name_or_path=base_model_name,
    tie_encoder_decoder=True
)

# --- 5. Train the Model ---
print("Starting unsupervised training (TSDAE)...")
print("This will take a long time (e.g., 1+ hour).")

# We will train for 1 epoch.
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=1,
    weight_decay=0,
    scheduler='constantlr',
    optimizer_params={'lr': 3e-5},
    show_progress_bar=True
)

# --- 6. Save the New, Fine-Tuned Model ---
output_model_path = './my_finetuned_model'
print(f"Training complete. Saving model to {output_model_path}")
model.save(output_model_path)
print("Script 1 complete. You can now run track_a.py.")