import os
import json
import pickle
import requests
from tqdm import tqdm

def download_alpaca_json(url, save_path):
   
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if os.path.exists(save_path):
        print(f"{save_path} already exists, skipping download.")
        return save_path

    print(f"Downloading from: {url}")
    response = requests.get(url, stream=True)
    total = int(response.headers.get('content-length', 0))

    with open(save_path, 'wb') as f, tqdm(
        desc=f"Downloading {os.path.basename(save_path)}",
        total=total,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))
    print(f"Downloaded to {save_path}")
    return save_path

def extract_alpaca_prompts(json_path):
    """prompt: instruction + input(if not empty)"""
    results = []
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in tqdm(data, desc="Extracting prompts"):
        instruction = item.get("instruction", "").strip()
        input_text = item.get("input", "").strip()
        if instruction:
            if input_text:
                prompt = f"{instruction}\n\n{input_text}"
            else:
                prompt = instruction
            results.append({"prompt": prompt})

    print(f"Extracted {len(results)} prompts")
    return results

def save_to_pickle(data, path):
    with open(path, "wb") as f:
        pickle.dump(data, f)
    print(f"Saved to {path}")

def preview_prompts(pkl_path, num=3):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    for i, item in enumerate(data[:num]):
        print(f"\n--- Prompt #{i+1} ---\n{item['prompt']}")

def main():
    url = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
    json_path = "data/alpaca_raw.json"
    output_path = "data/alpaca_prompts.pkl"

    download_alpaca_json(url, json_path)
    prompts = extract_alpaca_prompts(json_path)
    save_to_pickle(prompts, output_path)
    preview_prompts(output_path)

if __name__ == "__main__":
    main()
