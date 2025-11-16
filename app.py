from flask import Flask, render_template, request
from src.utils import (
    load_local_image,
    download_image,
    get_image_embedding,
    get_text_embedding,
    query_pinecone
)
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    results = []
    answer = None

    if request.method == "POST":
        mode = request.form.get("mode")

        if mode == "text_query":
            user_query = request.form.get("text_query")
            q_vec = get_text_embedding(user_query)
            pine_results = query_pinecone(q_vec)

        elif mode == "image_query":
            img_file = request.files["image_file"]
            path = "uploaded.jpg"
            img_file.save(path)

            img = load_local_image(path)
            q_vec = get_image_embedding(img)

            pine_results = query_pinecone(q_vec)

        # Format context for Gemini
        context = ""
        for m in pine_results:
            results.append(m.metadata)
            context += f"Text: {m.metadata['text']}\nImage URL: {m.metadata['image_url']}\n\n"

        # Gemini RAG response
        model = genai.GenerativeModel("gemini-2.0-flash")
        prompt = f"""
Use the retrieved context to answer the user's query.

Context:
{context}

User Query:
{user_query if mode == 'text_query' else 'Image query'}

Answer:
"""
        answer = model.generate_content(prompt).text

    return render_template("index.html", results=results, answer=answer)


if __name__ == "__main__":
    app.run(debug=True)
