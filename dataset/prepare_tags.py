"""
Script: Multi-GPU Tag Generation with InsTagger (via vLLM)

Overview:
This script performs multi-GPU batch tag generation using the InsTagger model.
It loads prompts from a `.pkl` file, generates tags with vLLM, and saves results.

Steps:
1. Setup: Configure logging, environment, and GPU assignment.
2. Load Data: Read input samples from a pickle file.
3. Prompt Preparation:
   - Each prompt is wrapped with an instruction prefix and suffix.
   - Tokenizer is used to measure token length.
   - Truncation is applied to ensure:
     [prefix + truncated_prompt + suffix] + [max_output_tokens] ≤ model's max length (e.g., 2048 tokens).
4. Inference (per GPU process):
   - Generate tags using vLLM.
   - Parse model output into tag list (max 5 per sample).
   - Track valid and invalid generations separately.
5. Save:
   - Periodically dump progress to disk.
   - After all processes finish, merge outputs into a final `.pkl` file.

Features:
✓ Token-aware truncation to prevent overflow  
✓ Parallel processing (multi-GPU)  
✓ Invalid result tracking  
✓ Safe periodic saving and merging
"""




import os
os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"
os.environ["VLLM_DISABLE_PROGRESS"] = "1"

import pickle
import json
import re
from tqdm import tqdm
from vllm import LLM, SamplingParams
from multiprocessing import Process
import torch

def extract_tags(text):
    text = text.strip()
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [d["tag"].strip() for d in data if isinstance(d, dict) and "tag" in d and d["tag"].strip()][:5]
        elif isinstance(data, dict) and "tag" in data:
            tag = data["tag"].strip()
            return [tag] if tag else []
    except Exception:
        pass
    matches = re.findall(r'"tag"\s*:\s*"([^"]+)"', text)
    return [tag.strip() for tag in matches if tag.strip()][:5]

def prepare_input_text(prompt, tokenizer, max_total_tokens=2048, max_output_tokens=150):
    base_prompt = (
        "You are a helpful assistant. Please identify tags of user intentions in the following user query "
        "and provide an explanation for each tag. Please respond in the JSON format "
        "{\"tag\": str, \"explanation\": str}.\n\nQuery: "
    )
    suffix = "\nAssistant:"
    base_tokens = tokenizer.encode(base_prompt + suffix, add_special_tokens=False)
    available_tokens = max_total_tokens - max_output_tokens - len(base_tokens)

    full_prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    truncated_prompt_tokens = full_prompt_tokens[:available_tokens]
    
    

    truncated_prompt = tokenizer.decode(truncated_prompt_tokens)
    return base_prompt + truncated_prompt + suffix


def worker_process(data_chunk, gpu_id, output_path, process_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    llm = LLM(
        model="OFA-Sys/InsTagger",
        dtype="float16",
        tensor_parallel_size=1,
        trust_remote_code=True,
        max_model_len=2048,
        gpu_memory_utilization=0.8,
        enforce_eager=True,
    )
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=150,
        stop=["</s>", "<|endoftext|>", "\n\n"]
    )

    results = []
    invalid_samples = []
    batch_size = 64
    error_count = successful_tags = 0

    for i in tqdm(range(0, len(data_chunk), batch_size), desc=f"GPU-{gpu_id} Processing", position=process_id):
        batch = data_chunk[i:i + batch_size]
        input_texts, valid_items = [], []

        for item in batch:
            prompt = item.get("prompt", "")
            try:
                input_text = prepare_input_text(prompt, tokenizer)
            
                input_texts.append(input_text)
                valid_items.append(item)
            except Exception:
                item["tags"] = []
                results.append(item)
                error_count += 1

        if input_texts:
            try:
                outputs = llm.generate(input_texts, sampling_params)
                for item, output in zip(valid_items, outputs):
                    generated_text = output.outputs[0].text
                    tags = extract_tags(generated_text)
                    item["tags"] = tags
                    if tags:
                        successful_tags += 1
                    else:
                        invalid_samples.append({
                            "prompt": item.get("prompt", ""),
                            "output": generated_text
                        })
                results.extend(valid_items)
            except Exception:
                error_count += 1
                for item in valid_items:
                    item["tags"] = []
                results.extend(valid_items)

        if len(results) % 128 == 0 and len(results) > 0:
            with open(output_path, "wb") as f:
                pickle.dump(results, f)
            print(f"GPU-{gpu_id} saves: {len(results)} items")

    with open(output_path, "wb") as f:
        pickle.dump(results, f)
    if invalid_samples:
        invalid_output_path = output_path.replace(".pkl", "_invalid.pkl")
        with open(invalid_output_path, "wb") as f:
            pickle.dump(invalid_samples, f)
        print(f"GPU-{gpu_id} saved {len(invalid_samples)} invalid samples to {invalid_output_path}")

    print(f"\nGPU-{gpu_id} completed: {len(results)} samples | Errors: {error_count} | Tagged: {successful_tags}")

def main():

    #change here to inference different dataset
    input_path = "extractors/data/pii_prompts.pkl"
    final_output = "extractors/data/pii_tagged_total.pkl"

    with open(input_path, "rb") as f:
        data = pickle.load(f)

    print(f" Loaded {len(data)} samples")
    num_gpus = 4  # torch.cuda.device_count()
    print(f" Using {num_gpus} GPUs")

    chunk_size = len(data) // num_gpus
    data_chunks = [data[i*chunk_size: (i+1)*chunk_size if i<num_gpus-1 else len(data)] for i in range(num_gpus)]

    processes, chunk_paths = [], []
    for i in range(num_gpus):
        chunk_path = f"{final_output.replace('.pkl', '')}_gpu{i}.pkl"
        chunk_paths.append(chunk_path)
        p = Process(target=worker_process, args=(data_chunks[i], i, chunk_path, i))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("Merging results...")
    all_results = []
    for path in chunk_paths:
        with open(path, "rb") as f:
            all_results.extend(pickle.load(f))

    with open(final_output, "wb") as f:
        pickle.dump(all_results, f)

    total_tagged = sum(1 for item in all_results if isinstance(item, dict) and item.get("tags"))
    print(f"Processing completed! {len(all_results)} samples processed, {total_tagged} tagged")
    print(f"Final merged file: {final_output}")

if __name__ == "__main__":
    main()
