import os
import requests
import numpy as np
import torch
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
from transformers import CLIPProcessor, CLIPModel
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

# -----------------------------
# Load environment variables
# -----------------------------
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# -----------------------------
# Load CLIP model (Free)
# -----------------------------
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

# -----------------------------
# Pinecone Initialize
# -----------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "image-text-embeddings"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=512,  # CLIP ViT-B/32
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)


# -----------------------------
# Utility Functions
# -----------------------------
def download_image(url):
    """Download an image from a URL."""
    try:
        res = requests.get(url, timeout=5)
        res.raise_for_status()
        return Image.open(BytesIO(res.content)).convert("RGB")
    except:
        return None


def load_local_image(path):
    """Load a local image file."""
    try:
        return Image.open(path).convert("RGB")
    except:
        return None


def get_image_embedding(image):
    """Generate CLIP image embedding."""
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        emb = model.get_image_features(**inputs)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.squeeze().numpy()


def get_text_embedding(text):
    """Generate CLIP text embedding."""
    inputs = processor(text=text, return_tensors="pt")
    with torch.no_grad():
        emb = model.get_text_features(**inputs)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.squeeze().numpy()


def combine_embeddings(image_vec, text_vec, w_img=0.5, w_txt=0.5):
    """Combine embeddings using weighted sum."""
    return w_img * image_vec + w_txt * text_vec


def query_pinecone(vector, top_k=5):
    """Search Pinecone using a vector."""
    return index.query(
        vector=vector.tolist(),
        top_k=top_k,
        include_metadata=True
    ).matches


def upsert_to_pinecone(id_value, vector, metadata):
    """Upsert vector into Pinecone."""
    index.upsert([{
        "id": id_value,
        "values": vector.tolist(),
        "metadata": metadata
    }])
