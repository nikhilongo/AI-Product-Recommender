"""
vector_convert.py
-----------------
Creates a multi-modal FAISS index for text + image embeddings.
"""

import os
import ast
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss
import torch
from PIL import Image
import requests
from io import BytesIO
import open_clip

# ========================
# CONFIG
# ========================
CSV_PATH = r"C:\Users\NIKHIL GUPTA\Desktop\ikarus\data\cleaned.csv"
TEXT_COLUMN = "merged_text"
IMAGE_COLUMN = "images"  # change if your CSV has a different column name

ARTIFACT_DIR = Path("./rag_artifacts")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
TEXT_INDEX_PATH = ARTIFACT_DIR / "faiss_text_index.bin"
IMAGE_INDEX_PATH = ARTIFACT_DIR / "faiss_image_index.bin"
META_PATH = ARTIFACT_DIR / "metadata.pkl"

TEXT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 256
# ========================

# Load models
print("Loading text model...")
text_model = SentenceTransformer(TEXT_MODEL_NAME)

print("Loading OpenCLIP model...")
clip_model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32', pretrained='laion400m_e32'
)
clip_model.eval()

# Helper functions
def clean_image_urls(url_str):
    try:
        urls = ast.literal_eval(url_str)
        return [u.strip() for u in urls if u.strip()]
    except:
        return []

def get_image_embedding(url):
    try:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        img_input = preprocess(img).unsqueeze(0)
        with torch.no_grad():
            emb = clip_model.encode_image(img_input)
            emb /= emb.norm(dim=-1, keepdim=True)
        return emb.squeeze(0).numpy()
    except:
        return None

# Load CSV
print(f"Loading CSV from {CSV_PATH}...")
df = pd.read_csv(CSV_PATH, usecols=[TEXT_COLUMN, IMAGE_COLUMN])
df[IMAGE_COLUMN] = df[IMAGE_COLUMN].fillna("")
df['cleaned_images'] = df[IMAGE_COLUMN].apply(clean_image_urls)
texts = df[TEXT_COLUMN].fillna("").tolist()

# Metadata
metadata = [{"id": str(i), "text": t} for i, t in enumerate(texts)]

# Create text embeddings
print("Creating text embeddings...")
text_embeddings = []
for i in tqdm(range(0, len(texts), BATCH_SIZE)):
    batch_texts = texts[i:i+BATCH_SIZE]
    emb = text_model.encode(batch_texts, convert_to_numpy=True)
    text_embeddings.append(emb)
text_embeddings = np.vstack(text_embeddings).astype("float32")
faiss.normalize_L2(text_embeddings)

# Create image embeddings
print("Creating image embeddings...")
image_embeddings = []
for urls in tqdm(df['cleaned_images']):
    embs = [get_image_embedding(url) for url in urls if get_image_embedding(url) is not None]
    if embs:
        avg_emb = np.mean(embs, axis=0)
        image_embeddings.append(avg_emb)
    else:
        image_embeddings.append(np.zeros(512))  # fallback
image_embeddings = np.array(image_embeddings).astype("float32")
faiss.normalize_L2(image_embeddings)

# Save FAISS indexes
print("Saving FAISS indexes...")
text_index = faiss.IndexFlatIP(text_embeddings.shape[1])
text_index.add(text_embeddings)
faiss.write_index(text_index, str(TEXT_INDEX_PATH))

image_index = faiss.IndexFlatIP(image_embeddings.shape[1])
image_index.add(image_embeddings)
faiss.write_index(image_index, str(IMAGE_INDEX_PATH))

# Save metadata
with open(META_PATH, "wb") as f:
    pickle.dump(metadata, f)

print(f"✅ Multi-modal embeddings saved in {ARTIFACT_DIR.absolute()}")
