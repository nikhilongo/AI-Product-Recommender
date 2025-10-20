import pandas as pd
import re

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)  # keep only letters, numbers, and spaces
    text = re.sub(r'\s+', ' ', text).strip()  # remove extra spaces
    return text

def clean_and_merge_csv(input_path, output_path="cleaned.csv"):
    # Load the CSV file
    df = pd.read_csv(input_path)
    
    # Columns to merge
    cols_to_merge = ['title', 'description', 'category', 'material', 'color']
    existing_cols = [col for col in cols_to_merge if col in df.columns]
    
    # Clean each column
    for col in existing_cols:
        df[col] = df[col].apply(clean_text)
    
    # Merge columns
    df["merged_text"] = df[existing_cols].agg(" ".join, axis=1)
    df["merged_text"] = df["merged_text"].apply(lambda x: re.sub(r'\s+', ' ', x).strip())
    
    # Remove empty and duplicate rows
    df = df[df["merged_text"].str.strip() != ""]
    df = df.drop_duplicates(subset=["merged_text"])
    
    # Save cleaned file
    df.to_csv(output_path, index=False)
    print(f"✅ Cleaned data saved to {output_path}")

if __name__ == "__main__":
    input_file = "C:/Users/NIKHIL GUPTA/Desktop/ikarus/data/intern_data_ikarus.csv"  # Change this path if needed
    clean_and_merge_csv(input_file)
