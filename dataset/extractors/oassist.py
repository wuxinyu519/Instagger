from datasets import load_dataset
import os
import pickle
from tqdm import tqdm

def extract_oasst_prompts(dataset_name="OpenAssistant/oasst1", split="train"):
    # messages load
    ds = load_dataset(dataset_name, split=split)
    # message_tree_id 
    convs = {}
    for ex in tqdm(ds, desc="Loading OASST1 messages"):
        if ex["role"] != "prompter":
            continue
        tree_id = ex["message_tree_id"]
        text = ex["text"].strip()
        if not text:
            continue
        convs.setdefault(tree_id, []).append(text)
    # merged prompts
    results = []
    for tree_id, turns in convs.items():
        merged = " ".join(turns)
        results.append({"prompt": merged})
    print(f"extract merged {len(results)}  prompts")
    return results

def save_to_pickle(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)
    print(f"save to {path}, in total {len(data)} items")

def preview(pkl_path, num=3):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    for i, item in enumerate(data[:num]):
        print(f"\n--- Prompt #{i+1} ---\n{item['prompt']}")

def main():
    prompts = extract_oasst_prompts(split="train")
    save_path = "extractors/data/oasst1_prompts.pkl"
    save_to_pickle(prompts, save_path)
    preview(save_path)

if __name__ == "__main__":
    main()
