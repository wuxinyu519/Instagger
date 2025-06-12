from datasets import load_dataset
import pickle
import os
from tqdm import tqdm

def extract_flan_prompts(dataset_name="Muennighoff/flan", split="train"):
    dataset = load_dataset(dataset_name, split=split, ignore_verifications=True)
    results = []

    for item in tqdm(dataset, desc="Extracting FLAN prompts"):
        prompt = item.get("inputs", "").strip()
        if prompt:
            results.append({"prompt": prompt})

    print(f"Extracted {len(results)} prompts from {dataset_name} ({split})")
    return results

def save_prompts(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)
    print(f"Saved to {path}")

def preview(path, num=3):
    with open(path, "rb") as f:
        data = pickle.load(f)
    for i, item in enumerate(data[:num]):
        print(f"\n--- Prompt #{i+1} ---\n{item['prompt']}")

def main():
    save_path = "data/flan_prompts.pkl"
    prompts = extract_flan_prompts()
    save_prompts(prompts, save_path)
    preview(save_path)

if __name__ == "__main__":
    main()
