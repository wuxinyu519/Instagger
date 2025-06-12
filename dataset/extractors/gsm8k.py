from datasets import load_dataset
import os
import pickle
from tqdm import tqdm

def extract_gsm8k_prompts(dataset_name="openai/gsm8k", config="main", split="train"):
    print(f"Loading GSM8K dataset: {dataset_name} ({config}/{split})...")
    ds = load_dataset(dataset_name, config, split=split)

    print(f"Extracting prompts from {len(ds)} examples...")
    results = []
    for item in tqdm(ds, desc="Extract GSM8K"):
        question = item.get("question", "").strip()
        if question:
            results.append({"prompt": question})
    print(f"Extracted {len(results)} prompts")
    return results

def save_prompts(prompts, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(prompts, f)
    print(f"Saved prompts to {output_path}")

def preview_prompts(pkl_path, num=5):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    for i, item in enumerate(data[:num]):
        print(f"\n--- Prompt #{i+1} ---")
        print(item["prompt"])

def main():
    prompts = extract_gsm8k_prompts()
    out_path = "extractors/data/gsm8k_prompts.pkl"
    save_prompts(prompts, out_path)
    preview_prompts(out_path, num=3)

if __name__ == "__main__":
    main()
