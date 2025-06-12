from datasets import load_dataset
import os
import pickle
from tqdm import tqdm

TOXICITY_FIELDS = [
    "toxicity",
    "severe_toxicity",
    "identity_attack",
    "insult",
    "threat",
    "profanity",
    "sexually_explicit",
    "flirtation"
]

def extract_toxicity_prompts(dataset_name="allenai/real-toxicity-prompts", split="train", threshold=0.3):
    print(f"Loading dataset: {dataset_name} ({split})...")
    ds = load_dataset(dataset_name, split=split)

    results = []
    for example in tqdm(ds, desc="Extracting prompt.text and sorted tags"):
        text = example.get("prompt", {}).get("text", "").strip()
        if not text:
            continue

        filtered = {}
        valid_score_found = False

        for field in TOXICITY_FIELDS:
            score = example.get("prompt", {}).get(field) 
            if score is None:
                continue
            valid_score_found = True
            if score > threshold: #only extract > 0.3
                filtered[field] = score

        if not valid_score_found:
            continue  # skip examples with all scores missing

        sorted_tags = sorted(filtered.items(), key=lambda x: x[1], reverse=True)
        tags = [tag for tag, _ in sorted_tags]

        results.append({
            "prompt": text,
            "tags": tags
        })


    print(f"Extracted {len(results)} prompts with sorted tags")
    return results

def save_prompts(prompts, output_path="./data/toxicity_prompts.pkl"):
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
    prompts = extract_toxicity_prompts()
    save_prompts(prompts)
    preview("./data/real_toxicity_prompts.pkl")

if __name__ == "__main__":
    main()
