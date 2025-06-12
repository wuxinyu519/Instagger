import os
import json
import pickle
import requests
from tqdm import tqdm

def download_file_from_url(url, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if os.path.exists(save_path):
        print(f"{save_path} already exists, skipping download.")
        return save_path

    response = requests.get(url, stream=True)
    response.raise_for_status()

    total = int(response.headers.get('content-length', 0))
    with open(save_path, 'wb') as file, tqdm(
        desc=f"Downloading {os.path.basename(save_path)}",
        total=total,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=8192):
            size = file.write(data)
            bar.update(size)

    print(f"Downloaded and saved to {save_path}")
    return save_path

def extract_prompts_from_json(path):
    results = []

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in tqdm(data, desc=f"Extracting from {os.path.basename(path)}"):
        conversations = item.get("conversations", [])
        human_turns = [turn.get("value", "").strip() for turn in conversations if turn.get("from") == "human" and turn.get("value", "").strip()]

        if human_turns:
            merged_prompt = " ".join(human_turns)
            results.append({"prompt": merged_prompt})

    print(f"{len(results)} valid prompts extracted from {os.path.basename(path)}")
    return results

def main():
    urls = [
        "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/HTML_cleaned_raw_dataset/sg_90k_part1_html_cleaned.json",
        "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/HTML_cleaned_raw_dataset/sg_90k_part2_html_cleaned.json",
    ]
    all_prompts = []

    for url in urls:
        fname = os.path.basename(url)
        json_path = download_file_from_url(url, os.path.join("data", fname))

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            print(f"{fname} contains {len(data)} conversations.")

        prompts = extract_prompts_from_json(json_path)
        all_prompts.extend(prompts)

    print(f"\n Total prompts extracted: {len(all_prompts)}")

    os.makedirs("data", exist_ok=True)
    with open("data/sharegpt_prompts.pkl", "wb") as f:
        pickle.dump(all_prompts, f)
    print(f"Saved to data/sharegpt_prompts.pkl")

if __name__ == "__main__":
    main()


    # import pickle
    # with open("data/sharegpt_prompts.pkl", "rb") as f:
    #     data = pickle.load(f)

    # # preview first 5 data
    # for i, item in enumerate(data[:5]):
    #     print(f"--- Prompt #{i+1} ---")
    #     print(item["prompt"])
    #     print()
