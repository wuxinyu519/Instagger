from datasets import load_dataset
import os
import pickle
from tqdm import tqdm

def extract_math_prompts(dataset_name="hendrycks/math", split="train"):
    print(f"Loading MATH dataset: {dataset_name} ({split})...")
    dataset = load_dataset(dataset_name, split=split)
    print(f"Extracting prompts from {len(dataset)} examples...")

    results = []
    for item in tqdm(dataset, desc="Extract MATH"):
        question = item.get("problem", "").strip()
        if question:
            results.append({"prompt": question})
    print(f"Extracted {len(results)} prompts")
    return results

def save_prompts(prompts, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(prompts, f)
    print(f"Saved prompts to {output_path}")

def preview_prompts(pkl_path, num=3):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    for i, item in enumerate(data[:num]):
        print(f"\n--- Prompt #{i+1} ---")
        print(item["prompt"])

def main():
    
    prompts = extract_math_prompts(split="train")
    out_path = "extractors/data/math_prompts.pkl"
    save_prompts(prompts, out_path)
    preview_prompts(out_path, num=3)

if __name__ == "__main__":
    main()
