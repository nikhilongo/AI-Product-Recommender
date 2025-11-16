import pandas as pd
import re

# Load your file
df = pd.read_csv("C:/Users/nikhi/Desktop/Projects/AI-Product-Recommender/data/intern_data_ikarus.csv")

# ---------------------------
# 1. Clean text columns
# ---------------------------

def clean_text(x):
    if pd.isna(x):
        return ""
    x = str(x)
    x = re.sub(r'\s+', ' ', x)            # remove extra spaces
    x = re.sub(r'[^\w\s,.-]', '', x)      # remove unwanted characters
    return x.strip()

cols_to_clean = ["title", "brand", "description", "price",
                 "categories", "material", "color"]

for col in cols_to_clean:
    if col in df.columns:
        df[col] = df[col].apply(clean_text)

# ---------------------------
# 2. Clean image URLs
# ---------------------------

def clean_image_url(url):
    if pd.isna(url):
        return ""
    url = str(url).strip()
    url = url.split()[0]                  # keep first URL if multiple
    url = re.sub(r'[^a-zA-Z0-9.:/_-]', '', url)  # remove junk
    return url

if "images" in df.columns:
    df["cleaned_images"] = df["images"].apply(clean_image_url)

# ---------------------------
# 3. Combine cleaned columns
# ---------------------------

df["combined_cleaned"] = (
    df["title"] + " | "
    + df["brand"] + " | "
#    + df["description"] + " | "
    + df["price"] + " | "
    + df["categories"] + " | "
    + df["material"] + " | "
    + df["color"]
).str.strip(" |")

# ---------------------------
# 4. Save output
# ---------------------------

df.to_csv("C:/Users/nikhi/Desktop/Projects/AI-Product-Recommender/data/cleaned_output.csv", index=False)

print("Cleaning complete! File saved as cleaned_output.csv.")
