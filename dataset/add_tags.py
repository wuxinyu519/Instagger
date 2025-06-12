import pickle
import os

def batch_add_tags(pkl_path, save_path, tags_to_add):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    print(f"Loaded {len(data)} prompts from {pkl_path}.")
    print(f"Adding tags: {tags_to_add}")

    for item in data:
        existing_tags = item.get("tags", [])
        # 合并并去重
        item["tags"] = list(dict.fromkeys(existing_tags + tags_to_add))

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(data, f)

    print(f"\nUpdated prompts saved to: {save_path}")

if __name__ == "__main__":
    file_name = "pii_tagged_total.pkl" 

    original_path = f"extractors/data/{file_name}"
    new_path = f"extractors/final_tagged_data/{file_name}"
    tags = ["privacy", "personally identifiable information"]

    batch_add_tags(original_path, new_path, tags)

