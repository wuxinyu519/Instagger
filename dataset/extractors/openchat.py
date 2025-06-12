import os
import json
import pickle
import requests
from tqdm import tqdm

def download_hf_lfs_file(hf_url, save_path):
    """Download large file hosted on Hugging Face with LFS redirection."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if os.path.exists(save_path):
        print(f"{save_path} already exists, skipping download.")
        return save_path

    print(f"Getting redirect URL for: {hf_url}")
    session = requests.Session()
    response = session.head(hf_url, allow_redirects=True)
    real_url = response.url  # Follow LFS redirect

    print(f"➡️ Redirected to: {real_url}")
    response = session.get(real_url, stream=True)
    total = int(response.headers.get('content-length', 0))

    with open(save_path, 'wb') as file, tqdm(
        desc=f"Downloading {os.path.basename(save_path)}",
        total=total,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)
                bar.update(len(chunk))

    print(f"Downloaded and saved to {save_path}")
    return save_path

def extract_prompts_from_json(path):
    results = []
    total_gpt_responses = 0

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in tqdm(data, desc=f"Extracting from {os.path.basename(path)}"):
        conversations = item.get("items", [])

        # count GPT responses
        total_gpt_responses += sum(
            1 for turn in conversations if turn.get("from") == "gpt" and turn.get("value", "").strip()
        )

        # extract human input
        human_turns = [
            turn["value"].strip() for turn in conversations
            if turn.get("from") == "human" and isinstance(turn.get("value"), str) and turn["value"].strip()
        ]

        if human_turns:
            merged_prompt = " ".join(human_turns)
            results.append({"prompt": merged_prompt})

    print(f"{len(results)} valid prompts extracted from {os.path.basename(path)}")
    print(f"Total GPT responses: {total_gpt_responses}")
    return results


def preview_sample_conversation(json_path):
    """Print one sample conversation and count total conversations."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"\nTotal conversations: {len(data)}")
    print("\nSample conversation (index 0):\n")
    print(json.dumps(data[0], indent=2, ensure_ascii=False))

def preview_prompts(pkl_path, num=5):
    """Print preview of the first few prompts in a .pkl file."""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    for i, item in enumerate(data[:num]):
        print(f"\n--- Prompt #{i+1} ---")
        print(item["prompt"])

def main():
    url = "https://huggingface.co/datasets/openchat/openchat_sharegpt4_dataset/resolve/main/sharegpt_gpt4.json"
    fname = os.path.basename(url)
    json_path = download_hf_lfs_file(url, os.path.join("data", fname))

    # preview_sample_conversation(json_path)

    prompts = extract_prompts_from_json(json_path)

    output_path = "data/openchat_prompts.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(prompts, f)
    print(f"\nSaved {len(prompts)} prompts to {output_path}")

    preview_prompts(output_path, num=5)

if __name__ == "__main__":
    main()
    
