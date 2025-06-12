import os
import pickle
import json
import re
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.multiprocessing import Process, set_start_method

# 加载模型
def load_instagger(device):
    model = AutoModelForCausalLM.from_pretrained(
        "OFA-Sys/InsTagger",
        torch_dtype=torch.float16,
        device_map={"": device},
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "OFA-Sys/InsTagger",
        use_fast=False,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

# 标签生成函数
def generate_tags(prompt_text, model, tokenizer, device):
    input_text = f"""You are a helpful assistant. Please identify tags of user intentions in the following user query and provide an explanation for each tag. Please respond in the JSON format {{"tag": str, "explanation": str}}.\n\nQuery: {prompt_text}\nAssistant:"""
    try:
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        input_len = inputs["input_ids"].shape[1]
        new_tokens = output[0][input_len:]
        tags_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        return extract_tags(tags_text)
    except Exception as e:
        print(f"Error: {e}")
        return []

# get generated tags
def extract_tags(tags_text):
    try:
        if tags_text.strip().startswith('['):
            tags_data = json.loads(tags_text)
            return [item["tag"] for item in tags_data if isinstance(item, dict) and "tag" in item][:5]
    except:
        pass
    return re.findall(r'"tag"\s*:\s*"([^"]+)"', tags_text)[:5]

# work_process
def worker(data_chunk, device_id, save_path):
    device = f"cuda:{device_id}"
    model, tokenizer = load_instagger(device)

    results = []
    for idx, row in tqdm(data_chunk.iterrows(), total=len(data_chunk), desc=f"GPU{device_id} tagging"):
        prompt = str(row["prompt"])
        tags = generate_tags(prompt, model, tokenizer, device)
        results.append({"prompt": prompt, "tags": tags})

        if (idx + 1) % 100 == 0 or (idx + 1) == len(data_chunk):
            with open(save_path, "wb") as f:
                pickle.dump(results, f)
            print(f"GPU{device_id} saved {idx + 1} samples to {save_path}")

# parallel
def main():
    set_start_method("spawn", force=True)  # multi processes
    root_path = "extractors/data"
    input_path = f"{root_path}/routerbench_0shot.pkl"

    with open(input_path, "rb") as f:
        full_data = pickle.load(f)

    num_gpus = torch.cuda.device_count()
    # num_gpus = 7
    chunk_size = len(full_data) // num_gpus
    chunks = [full_data.iloc[i * chunk_size: (i + 1) * chunk_size] for i in range(num_gpus)]
    if len(full_data) % num_gpus != 0:
        chunks[-1] = pd.concat([chunks[-1], full_data.iloc[num_gpus * chunk_size:]])

    print(f"Loaded {len(full_data)} samples.")
    print(f"Using {num_gpus} GPUs...")

    processes = []
    for i in range(num_gpus):
        save_path = f"{root_path}/routerbench_tagged_gpu{i}.pkl"
        p = Process(target=worker, args=(chunks[i], i, save_path))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print(" All processes completed.")

if __name__ == "__main__":
    main()
