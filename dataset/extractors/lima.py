from datasets import load_dataset
import os
import pickle
from tqdm import tqdm
from huggingface_hub import HfFolder

def extract_lima_prompts(dataset_name="GAIR/lima", split="train"):
    token = HfFolder.get_token()  # Needs token
    print(f"Loading dataset: {dataset_name} ({split})...")
    ds = load_dataset(dataset_name, split=split, token=token)

    
    for i in range(3):
        print(f"\n=== Example {i+1} ===")
        for k, v in ds[i].items():
            print(f"{k}: {v if isinstance(v, str) else v}")

    results = []
    for item in tqdm(ds, desc="Extracting user prompts"):
        conv = item.get("conversations", [])
        user_turns = [conv[i].strip() for i in range(0, len(conv), 2)]  # even index are prompter content
        if user_turns:
            merged_prompt = "\n".join(user_turns)
            results.append({"prompt": merged_prompt})
    print(f"Extracted {len(results)} prompts")
    return results

def save_prompts(prompts, output_path="extractors/data/lima_prompts.pkl"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(prompts, f)
    print(f"Saved to {output_path}")

def preview_prompts(path, num=3):
    with open(path, "rb") as f:
        data = pickle.load(f)
    for i, item in enumerate(data[:num]):
        print(f"\n--- Prompt #{i+1} ---\n{item['prompt']}")

def main():
    prompts = extract_lima_prompts()
    save_prompts(prompts)
    preview_prompts("extractors/data/lima_prompts.pkl")

if __name__ == "__main__":
    main()
