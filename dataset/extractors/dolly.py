import os
import json
import pickle
from tqdm import tqdm
import requests

DOLLY_URL = "https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl"

def download_dolly_jsonl(url, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if os.path.exists(save_path):
        print(f"{save_path} already exists, skipping download.")
        return save_path

    print(f"Downloading from: {url}")
    response = requests.get(url, stream=True)
    with open(save_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print(f" Downloaded to {save_path}")
    return save_path

def extract_dolly_prompts(jsonl_path):
    results = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Extracting prompts"):
            item = json.loads(line)
            instruction = item.get("instruction", "").strip()
            context = item.get("context", "").strip()
            if instruction:
                prompt = instruction if not context else f"{instruction}\n\n{context}"
                results.append({"prompt": prompt})
    print(f"Extracted {len(results)} prompts")
    return results

def save_to_pickle(data, output_path):
    with open(output_path, "wb") as f:
        pickle.dump(data, f)
    print(f"Saved to {output_path}")

def preview_prompts(pkl_path, num=3):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    for i, item in enumerate(data[:num]):
        print(f"\n--- Prompt #{i+1} ---\n{item['prompt']}")

def main():
    jsonl_path = "data/dolly_raw.jsonl"
    output_path = "data/dolly_prompts.pkl"

    download_dolly_jsonl(DOLLY_URL, jsonl_path)
    prompts = extract_dolly_prompts(jsonl_path)
    save_to_pickle(prompts, output_path)
    preview_prompts(output_path)

if __name__ == "__main__":
    main()
