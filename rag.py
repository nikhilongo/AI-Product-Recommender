"""
rag_multimodal.py
-----------------
RAG chatbot with text + image retrieval support.
"""

import os
import pickle
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import faiss
import google.generativeai as genai
import torch
from PIL import Image
import requests
from io import BytesIO
import open_clip

# ========================
# CONFIG
# ========================
ARTIFACT_DIR = Path("./rag_artifacts")
TEXT_INDEX_PATH = ARTIFACT_DIR / "faiss_text_index.bin"
IMAGE_INDEX_PATH = ARTIFACT_DIR / "faiss_image_index.bin"
META_PATH = ARTIFACT_DIR / "metadata.pkl"
TEXT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5
# ========================

# Load environment
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env")
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")
chat = model.start_chat(history=[])

# Load FAISS indexes and metadata
text_index = faiss.read_index(str(TEXT_INDEX_PATH))
image_index = faiss.read_index(str(IMAGE_INDEX_PATH))
with open(META_PATH, "rb") as f:
    metadata = pickle.load(f)

text_model = SentenceTransformer(TEXT_MODEL_NAME)
clip_model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32', pretrained='laion400m_e32'
)
clip_model.eval()

# Helper functions
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

def retrieve_hybrid(text_query=None, image_url=None, k=TOP_K, text_weight=0.5, image_weight=0.5):
    scores = {}
    if text_query:
        q_emb = text_model.encode([text_query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(q_emb)
        D, I = text_index.search(q_emb, k*2)
        for score, idx in zip(D[0], I[0]):
            scores[idx] = scores.get(idx, 0) + text_weight*score
    if image_url:
        q_emb = get_image_embedding(image_url).astype("float32").reshape(1,-1)
        faiss.normalize_L2(q_emb)
        D, I = image_index.search(q_emb, k*2)
        for score, idx in zip(D[0], I[0]):
            scores[idx] = scores.get(idx, 0) + image_weight*score

    top_idx = sorted(scores, key=lambda x: scores[x], reverse=True)[:k]
    return [metadata[i]['text'] for i in top_idx]

def build_prompt(context_chunks, user_query):
    context_text = "\n\n".join([f"[Context {i+1}]: {chunk}" for i, chunk in enumerate(context_chunks)])
    prompt = (
        "You are a knowledgeable assistant. Use the following context to answer the user's question. "
        "If the answer is not in the context, say so politely.\n\n"
        f"{context_text}\n\nUser question: {user_query}"
    )
    return prompt

# Interactive chatbot
print("🤖 Multi-Modal RAG Chatbot (type 'exit' to quit)")
while True:
    user_input = input("\nYou (text): ").strip()
    if user_input.lower() in ["exit", "quit"]:
        print("👋 Goodbye!")
        break
    user_image = input("Image URL (optional, leave blank if none): ").strip() or None

    context_chunks = retrieve_hybrid(text_query=user_input, image_url=user_image)
    full_prompt = build_prompt(context_chunks, user_input)
    
    try:
        response = chat.send_message(full_prompt)
        print("\nGemini:", response.text)
    except Exception as e:
        print("❌ Error:", e)
