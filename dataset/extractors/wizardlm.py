from datasets import load_dataset
import os
import json
import pickle
from tqdm import tqdm

def extract_wizardlm_prompts(dataset_name="WizardLM/WizardLM_evol_instruct_V2_196k", split="train"):
    print(f"Loading dataset: {dataset_name} ({split})...")
    dataset = load_dataset(dataset_name, split=split)

    print("\nSample:")
    print(json.dumps(dataset[0], indent=2, ensure_ascii=False))

    results = []
    for item in tqdm(dataset, desc="🔍 Extracting prompts"):
        conversations = item.get("conversations", [])
        human_turns = [
            turn["value"].strip()
            for turn in conversations
            if turn.get("from") == "human" and isinstance(turn.get("value"), str)
        ]
        if human_turns:
            merged_prompt = " ".join(human_turns)
            results.append({"prompt": merged_prompt})

    print(f"\nExtracted {len(results)} prompts")
    return results

def save_to_pickle(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)
    print(f"Saved to {path}")

def preview_prompts(pkl_path, num=3):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    for i, item in enumerate(data[:num]):
        print(f"\n--- Prompt #{i+1} ---\n{item['prompt']}")

def main():
    save_path = "extractors/data/wizardlm_prompts.pkl"
    prompts = extract_wizardlm_prompts()
    save_to_pickle(prompts, save_path)
    preview_prompts(save_path)

if __name__ == "__main__":
    main()
