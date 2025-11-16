import pandas as pd
from src.utils import (
    download_image,
    get_image_embedding,
    get_text_embedding,
    combine_embeddings,
    upsert_to_pinecone
)


def process_csv_and_store(csv_path):
    df = pd.read_csv(csv_path)

    for idx, row in df.iterrows():
        image_url = row["cleaned_images"]
        text = row["combined_cleaned"]

        img = download_image(image_url)
        if img is None:
            print(f"Skipping row {idx}: image failed")
            continue

        image_vec = get_image_embedding(img)
        text_vec = get_text_embedding(text)

        combined_vec = combine_embeddings(image_vec, text_vec)

        metadata = {
            "image_url": image_url,
            "text": text
        }

        upsert_to_pinecone(f"item-{idx}", combined_vec, metadata)

    print("Completed embedding + Pinecone upload.")
