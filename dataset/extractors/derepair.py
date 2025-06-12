import os
import zipfile
import pickle
import requests
from tqdm import tqdm

DRREPAIR_URLS = {
    "codeforce": "https://nlp.stanford.edu/projects/myasu/DrRepair/raw_data/codeforce_data.zip",
    "deepfix": "https://nlp.stanford.edu/projects/myasu/DrRepair/raw_data/deepfix_data.zip",
    "spoc": "https://nlp.stanford.edu/projects/myasu/DrRepair/raw_data/spoc_data.zip"
}

DATA_DIR = "extractors/data/drrepair_raw"
OUTPUT_DIR = "extractors/data"

def download_and_unzip(name, url, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    zip_path = os.path.join(dest_dir, f"{name}.zip")

    if not os.path.exists(zip_path):
        print(f"Downloading {name}...")
        r = requests.get(url, stream=True)
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print(f"Downloaded: {zip_path}")
    else:
        print(f"Found existing zip: {zip_path}")

    extract_dir = os.path.join(dest_dir, name)
    if not os.path.exists(extract_dir):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print(f"Extracted to: {extract_dir}")
    return extract_dir

def collect_code_files(root):
    code_texts = []
    for dirpath, _, filenames in os.walk(root):
        for file in filenames:
            if file.endswith(".c") or file.endswith(".cpp") or file.endswith(".txt"):
                full_path = os.path.join(dirpath, file)
                try:
                    with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                        code = f.read().strip()
                        if len(code) > 10:
                            code_texts.append({"prompt": code})
                except Exception as e:
                    print(f"⚠️ Failed to read {full_path}: {e}")
    return code_texts

def save_pickle(data, name):
    output_path = os.path.join(OUTPUT_DIR, f"drrepair_{name}_prompts.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(data, f)
    print(f"Saved {len(data)} prompts to {output_path}")

def main():
    for name, url in DRREPAIR_URLS.items():
        folder = download_and_unzip(name, url, DATA_DIR)
        prompts = collect_code_files(folder)
        save_pickle(prompts, name)

if __name__ == "__main__":
    main()
