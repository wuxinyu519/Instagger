from datasets import load_dataset
import pickle
from tqdm import tqdm
import os

def extract_mbpp_prompts():
    print("Loading MBPP dataset (train)...")
    ds = load_dataset("mbpp", split="train")
    for i in range(3):
        print(f"\n--- Sample #{i+1} ---")
        for key, value in ds[i].items():
            print(f"{key}: {repr(value)}")
    prompts = []
    for item in tqdm(ds, desc="Extracting prompts"):
        prompt = item.get("text", "").strip()
        if prompt:
            prompts.append({"prompt": prompt})
    print(f"Extracted {len(prompts)} prompts")
    return prompts

def save_prompts(prompts, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(prompts, f)
    print(f"Saved prompts to {output_path}")

def preview_prompts(pkl_path, num=3):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    for i, item in enumerate(data[:num]):
        print(f"\n--- Prompt #{i+1} ---\n{item['prompt']}")

def main():
    output_path = "extractors/data/mbpp_prompts.pkl"
    prompts = extract_mbpp_prompts()
    save_prompts(prompts, output_path)
    preview_prompts(output_path)

if __name__ == "__main__":
    main()
