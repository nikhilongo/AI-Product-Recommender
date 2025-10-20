# backend/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle, faiss, numpy as np, torch, os, json, requests
from sentence_transformers import SentenceTransformer
from pathlib import Path
import open_clip
from PIL import Image
from io import BytesIO
import google.generativeai as genai
from dotenv import load_dotenv

# ==============================
# CONFIG
# ==============================
ARTIFACT_DIR = Path("./rag_artifacts")
TEXT_INDEX_PATH = ARTIFACT_DIR / "faiss_text_index.bin"
IMAGE_INDEX_PATH = ARTIFACT_DIR / "faiss_image_index.bin"
META_PATH = ARTIFACT_DIR / "metadata.pkl"
TEXT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5

# ==============================
# LOAD MODELS
# ==============================
print("🔹 Loading models and indexes...")
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("⚠️ GEMINI_API_KEY not found in .env")

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")

text_index = faiss.read_index(str(TEXT_INDEX_PATH))
image_index = faiss.read_index(str(IMAGE_INDEX_PATH))
with open(META_PATH, "rb") as f:
    metadata = pickle.load(f)

text_model = SentenceTransformer(TEXT_MODEL_NAME)
clip_model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion400m_e32"
)
clip_model.eval()

# ==============================
# FASTAPI APP
# ==============================
app = FastAPI(title="Furniture Recommendation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for dev; restrict in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================
# HELPERS
# ==============================
def get_image_embedding(url):
    try:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        img_input = preprocess(img).unsqueeze(0)
        with torch.no_grad():
            emb = clip_model.encode_image(img_input)
            emb /= emb.norm(dim=-1, keepdim=True)
        return emb.squeeze(0).numpy()
    except Exception as e:
        print("⚠️ Image embedding failed:", e)
        return None


def retrieve_products(query: str = None, image_url: str = None, k=TOP_K):
    scores = {}
    if query:
        q_emb = text_model.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(q_emb)
        D, I = text_index.search(q_emb, k * 2)
        for score, idx in zip(D[0], I[0]):
            scores[idx] = scores.get(idx, 0) + 0.5 * score
    if image_url:
        q_emb = get_image_embedding(image_url)
        if q_emb is not None:
            q_emb = q_emb.astype("float32").reshape(1, -1)
            faiss.normalize_L2(q_emb)
            D, I = image_index.search(q_emb, k * 2)
            for score, idx in zip(D[0], I[0]):
                scores[idx] = scores.get(idx, 0) + 0.5 * score
    top_idx = sorted(scores, key=lambda x: scores[x], reverse=True)[:k]
    return [metadata[i] for i in top_idx]


def generate_structured_recommendations(context_text, query):
    context_combined = "\n\n".join([c["text"] for c in context_text])
    prompt = f"""
    You are an expert furniture stylist and AI assistant.

    Based on the following product data:
    {context_combined}

    The user searched for: "{query}".

    Generate a JSON object with:
    {{
      "hero": "a short cozy one-line intro",
      "recommendations": [
        {{
          "title": "product name",
          "short": "1-sentence appealing summary",
          "features": ["3 bullet points of highlights"],
          "dimensions": "include if available",
          "price": "approx price or 'N/A'",
          "cta": "short call to action like 'Buy now' or 'Perfect for your space!'"
        }}
      ]
    }}

    Output only valid JSON, nothing else.
    """

    response = model.generate_content(prompt)
    text = response.candidates[0].content.parts[0].text.strip()

    # attempt to extract JSON safely
    try:
        json_text = text[text.find("{") : text.rfind("}") + 1]
        data = json.loads(json_text)
        return data
    except Exception as e:
        print("⚠️ JSON parsing failed:", e)
        return {"hero": "Couldn't generate recommendations", "recommendations": []}


# ==============================
# ENDPOINTS
# ==============================
class RecommendRequest(BaseModel):
    query: str
    image_url: str | None = None


@app.post("/recommend")
async def recommend(req: RecommendRequest):
    products = retrieve_products(req.query, req.image_url)
    data = generate_structured_recommendations(products, req.query)
    return data


@app.get("/")
async def home():
    return {"message": "Furniture Recommendation API is running 🚀"}
