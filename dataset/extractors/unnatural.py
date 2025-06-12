from datasets import load_dataset
import os
import pickle
from tqdm import tqdm

def extract_unnatural_prompts(dataset_name="mrm8488/unnatural-instructions-full", split="train"):
    print(f"Loading dataset: {dataset_name} ({split})...")
    ds = load_dataset(dataset_name, split=split)
    
    results = []
    for example in tqdm(ds, desc="🔍Extracting prompts"):
        instr = example.get("instruction", "").strip()
        inp = example.get("input", "").strip()
        if instr:
            if inp:
                prompt = f"{instr}\n\n{inp}"
            else:
                prompt = instr
            results.append({"prompt": prompt})
    print(f"Extracted {len(results)} prompts")
    return results

def save_prompts(prompts, output_path="extractors/data/unnatural_prompts.pkl"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(prompts, f)
    print(f"Saved to {output_path}")

def preview(prompts_path, num=3):
    with open(prompts_path, "rb") as f:
        data = pickle.load(f)
    for i, item in enumerate(data[:num]):
        print(f"\n--- Prompt #{i+1} ---\n{item['prompt']}")

def main():
    prompts = extract_unnatural_prompts()
    save_prompts(prompts)
    preview("extractors/data/unnatural_prompts.pkl")

if __name__ == "__main__":
    main()
