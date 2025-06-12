from datasets import load_dataset
import os
import pickle
from tqdm import tqdm

def extract_pli_prompts(dataset_name="ai4privacy/pii-masking-300k", split="train"):
    print(f"Loading dataset: {dataset_name} ({split})...")
    ds = load_dataset(dataset_name, split=split)

    results = []
    for example in tqdm(ds, desc="Extracting source_text prompts"):
        masked_text = example.get("source_text", "").strip()
        if masked_text:
            results.append({"prompt": masked_text, "tags": []})
    
    print(f"Extracted {len(results)} prompts")
    return results

def save_prompts(prompts, output_path="./data/pii_prompts.pkl"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(prompts, f)
    print(f"Saved to {output_path}")

def preview(prompts_path, num=3):
    with open(prompts_path, "rb") as f:
        data = pickle.load(f)
    for i, item in enumerate(data[:num]):
        print(f"\n--- Prompt #{i+1} ---\n{item['prompt']}\nTags: {item['tags']}")

def main():
    prompts = extract_pli_prompts()
    save_prompts(prompts)
    preview("data/pii_prompts.pkl")

if __name__ == "__main__":
    main()
