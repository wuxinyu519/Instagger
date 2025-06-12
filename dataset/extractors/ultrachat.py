from datasets import load_dataset
import os
import json
import pickle
from tqdm import tqdm

def extract_ultrachat_prompts(dataset_name="stingning/ultrachat", split="train"):
    print(f"Loading UltraChat dataset: {dataset_name} ({split})...")
    dataset = load_dataset(dataset_name, split=split)

    # sample
    print("\n Sample Conversation:")
    print(json.dumps(dataset[0], indent=2, ensure_ascii=False))

    results = []
    total_user_questions = 0

    for item in tqdm(dataset, desc="Extracting"):
        dialog = item.get("data", [])
        # user prompt
        user_turns = [dialog[i].strip() for i in range(0, len(dialog), 2) if dialog[i].strip()]
        
        if user_turns:
            merged_prompt = " ".join(user_turns)
            results.append({"prompt": merged_prompt})
            total_user_questions += len(user_turns)

    print(f"\nExtracted {len(results)} conversations.")
    print(f"Total user questions extracted: {total_user_questions}")
    return results

def save_prompts_to_pickle(prompts, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(prompts, f)
    print(f"\nSaved to {output_path}")

def preview_prompts(pkl_path, num=1):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    for i, item in enumerate(data[:num]):
        print(f"\n--- Prompt #{i+1} ---")
        print(item["prompt"])

def main():
    prompts = extract_ultrachat_prompts()
    save_path = "data/ultrachat_prompts.pkl"
    save_prompts_to_pickle(prompts, save_path)
    preview_prompts(save_path)

if __name__ == "__main__":
    main()
