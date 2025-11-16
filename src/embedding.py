import os
import requests
from io import BytesIO
from PIL import Image
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoProcessor, AutoModel


# ============================
# CONFIG
# ============================
CSV_PATH = "data/cleaned_output.csv"
IMAGE_DIR = "data/downloaded_images"
EMBED_SAVE_PATH = "product_embeddings.npy"

os.makedirs(IMAGE_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load SigLIP
processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
model = AutoModel.from_pretrained("google/siglip-base-patch16-224").to(device)


# ============================
# CLEAN URL
# ============================
def clean_url(url):
    if pd.isna(url):
        return ""
    url = str(url).strip().split()[0]
    return "".join(c for c in url if c.isalnum() or c in ":/.?&=_-")


# ============================
# SAFE IMAGE DOWNLOADER
# ============================
def download_image(url, save_path):
    try:
        resp = requests.get(url, timeout=10, stream=True)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert("RGB")
        img.save(save_path)
        return True
    except:
        return False


# ============================
# SAFE IMAGE LOADER (handles corruption)
# ============================
def load_image_safe(path):
    try:
        img = Image.open(path)
        img.verify()  # check if broken
        img = Image.open(path).convert("RGB")  # reopen properly
        return img
    except:
        return None


# ============================
# SIGLIP EMBEDDING FUNCTION
# ============================
def get_embeddings(image_path, text):
    # ---------- CASE 1: Try loading the image safely ----------
    img = None
    if image_path and os.path.exists(image_path):
        img = load_image_safe(image_path)

    # ---------- CASE 2: TEXT ONLY (if image missing/corrupt) ----------
    def embed_text_only():
        inputs = processor(
            text=[text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        text_vec = outputs.text_embeds
        text_vec = text_vec / text_vec.norm(dim=-1, keepdim=True)

        image_vec = torch.zeros_like(text_vec)
        combined = 0.7 * image_vec + 0.3 * text_vec

        return (
            image_vec.cpu().numpy(),
            text_vec.cpu().numpy(),
            combined.cpu().numpy()
        )

    # If no valid image → skip image processing
    if img is None:
        return embed_text_only()

    # ---------- CASE 3: Try preprocessing image ----------
    try:
        processed = processor(
            text=[text],
            images=[img],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64
        ).to(device)

        # If pixel_values missing → fallback
        if processed.get("pixel_values") is None:
            return embed_text_only()

    except Exception:
        return embed_text_only()

    # ---------- CASE 4: Safe model forward ----------
    with torch.no_grad():
        outputs = model(**processed)

    text_vec = outputs.text_embeds
    image_vec = outputs.image_embeds

    # Normalize
    text_vec = text_vec / text_vec.norm(dim=-1, keepdim=True)
    image_vec = image_vec / image_vec.norm(dim=-1, keepdim=True)

    combined = 0.7 * image_vec + 0.3 * text_vec

    return (
        image_vec.cpu().numpy(),
        text_vec.cpu().numpy(),
        combined.cpu().numpy()
    )


# ============================
# PROCESS CSV
# ============================
df = pd.read_csv(CSV_PATH)

all_vectors = []

for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing"):

    url = clean_url(row["cleaned_images"])
    text = str(row["combined_cleaned"])

    img_path = os.path.join(IMAGE_DIR, f"{idx}.jpg")

    # download image
    downloaded = download_image(url, img_path) if url else False

    if not downloaded:
        img_path = None

    # embed
    image_vec, text_vec, combined_vec = get_embeddings(img_path, text)

    all_vectors.append({
        "index": idx,
        "image_vec": image_vec.flatten(),
        "text_vec": text_vec.flatten(),
        "combined_vec": combined_vec.flatten()
    })


# ============================
# SAVE RESULTS
# ============================
np.save(EMBED_SAVE_PATH, all_vectors, allow_pickle=True)

print("\n✅ Done! SigLIP embeddings saved to:", EMBED_SAVE_PATH)
