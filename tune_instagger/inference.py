#!/usr/bin/env python3
"""
增量保存版本的InsTagger推理脚本 - 避免数据丢失
"""

import os
import pickle
import glob
from tqdm import tqdm
import argparse
from transformers import AutoTokenizer
from vllm import SamplingParams, LLM
import time

def find_latest_checkpoint(training_dir: str):
    """找到最新的checkpoint"""
    checkpoint_dirs = glob.glob(os.path.join(training_dir, "checkpoint-*"))
    if not checkpoint_dirs:
        return None
    
    checkpoint_nums = []
    for ckpt_dir in checkpoint_dirs:
        try:
            num = int(os.path.basename(ckpt_dir).split('-')[1])
            checkpoint_nums.append((num, ckpt_dir))
        except:
            continue
    
    if checkpoint_nums:
        latest = max(checkpoint_nums, key=lambda x: x[0])
        return latest[1]
    return None


def load_instagger_vllm(model_path):
    """使用 vLLM 加载模型和 tokenizer"""
    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        disable_log_stats=True,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=False,
        trust_remote_code=True,
        legacy=False
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return llm, tokenizer


def extract_tags(tags_text):
    """提取标签 - 简单按逗号分割，取前5个"""
    try:
        tags = [tag.strip() for tag in tags_text.split(',')]
        tags = [tag for tag in tags if tag]  
        return tags[:5]  
    except Exception as e:
        print(f"Error extracting tags: {e}")
        return []


def append_results_to_file(output_file, new_results):
    """增量追加结果到文件"""
    if not new_results:
        return 0
    
    existing_results = []
    if os.path.exists(output_file):
        try:
            with open(output_file, 'rb') as f:
                existing_results = pickle.load(f)
        except Exception as e:
            print(f"Warning: Could not load existing results, starting fresh: {e}")
            existing_results = []
    

    all_results = existing_results + new_results
    

    try:
        with open(output_file, 'wb') as f:
            pickle.dump(all_results, f)
        print(f"Saved {len(new_results)} new results. Total: {len(all_results)} samples")
        return len(all_results)
    except Exception as e:
        print(f"Error saving results: {e}")
        return len(existing_results)


def run_inference_vllm(model_path: str, data: list, output_file: str,
                       save_raw_text=True, save_interval=64, batch_size=32):
    
    print("Loading model...")
    llm, tokenizer = load_instagger_vllm(model_path)
    print("Model loaded successfully")

    print(f"Running vLLM inference on {len(data)} samples (batch_size={batch_size})...")
    
  
    initial_count = 0
    if os.path.exists(output_file):
        try:
            with open(output_file, 'rb') as f:
                existing_results = pickle.load(f)
                initial_count = len(existing_results)
            print(f"Found existing results file with {initial_count} samples")
        except:
            print("Starting with empty results file")
    
    start_time = time.time()
    batch_results = [] 
    total_processed = 0
    
    for start in tqdm(range(0, len(data), batch_size), desc="Processing batches"):
        batch = data[start:start + batch_size]
        prompts = [item['prompt'] for item in batch]
        input_texts = [
            f"You are a helpful assistant. Please identify tags of user intentions in the following user query and provide an explanation for each tag. Query: {p} Assistant: "
            for p in prompts
        ]

        sampling_params = SamplingParams(
            temperature=0.9,
            top_p=0.6,
            max_tokens=256,
            
        )

        try:
            responses = llm.generate(input_texts, sampling_params)
        except Exception as e:
            print(f"Error during vLLM batch generation: {e}")
            responses = [None] * len(batch)

        # 处理这个批次的结果
        current_batch_results = []
        for i, item in enumerate(batch):
            idx = start + i
            prompt = item['prompt']
            try:
                if responses[i] is not None:
                    raw_text = responses[i].outputs[0].text.strip()
                    # 去掉前缀 prompt 部分（如果有）
                    prefix = input_texts[i]
                    if raw_text.startswith(prefix):
                        generated_part = raw_text[len(prefix):].strip()
                    else:
                        generated_part = raw_text

                    tags = extract_tags(generated_part)
                else:
                    raw_text = "Generation failed"
                    tags = []

                result = {
                    'index': idx + 1,
                    'prompt': prompt,
                    'tags': tags,
                    'raw_tags': tags,
                    'original_item': item['original_item']
                }

                if save_raw_text:
                    result['raw_response'] = raw_text
                    result['full_input'] = input_texts[i]

                current_batch_results.append(result)

            except Exception as e:
                print(f"Error processing sample {idx+1}: {e}")
                result = {
                    'index': idx + 1,
                    'prompt': prompt,
                    'tags': [],
                    'error': str(e),
                    'original_item': item['original_item']
                }
                if save_raw_text:
                    result['raw_response'] = f"Error: {str(e)}"
                    result['full_input'] = "Error occurred"
                current_batch_results.append(result)

       
        batch_results.extend(current_batch_results)
        total_processed += len(current_batch_results)
        
        
        if len(batch_results) >= save_interval or (start + batch_size) >= len(data):
            total_saved = append_results_to_file(output_file, batch_results)
            
            print(f"Progress: {total_processed}/{len(data)} processed, {total_saved} total saved")
            
           
            batch_results = []

    end_time = time.time()
    print(f"Inference took {end_time - start_time:.2f} seconds")
    

    try:
        with open(output_file, 'rb') as f:
            final_results = pickle.load(f)
            print(f"Final verification: {len(final_results)} samples in output file")
        
       
        print("="*80)
        print("INFERENCE RESULTS PREVIEW")
        print("="*80)

        success_count = sum(1 for r in final_results if r.get('tags', []))
        print(f"Successfully processed: {success_count}/{len(final_results)} samples")

        
        new_results_start = max(0, len(final_results) - total_processed)
        preview_results = final_results[new_results_start:new_results_start + 3]
        
        for i, result in enumerate(preview_results):
            print(f"\n--- Sample {new_results_start + i + 1} ---")
            print(f"Prompt: {result['prompt'][:100]}...")
            print(f"Generated Tags: {result.get('tags', [])}")
            if save_raw_text and 'raw_response' in result:
                print(f"Raw Response: {result['raw_response'][:200]}...")
            print("-" * 60)

        print("="*80)
        return final_results
        
    except Exception as e:
        print(f"Warning: Could not verify final results: {e}")
        return []


def load_input_data(data_file: str, num_samples: int = 100):
    """从data.pkl文件加载数据"""
    try:
        if not os.path.exists(data_file):
            print(f"Error: {data_file} file not found!")
            return []
        
        with open(data_file, 'rb') as f:
            raw_data = pickle.load(f)
        
        print(f"Loaded {len(raw_data)} samples from {data_file}")
        
        if num_samples is not None and num_samples > 0:
            raw_data = raw_data[:num_samples]
            print(f"Limited to first {num_samples} samples")
        
        data = []
        for i, item in enumerate(raw_data):
            prompt = None
            if isinstance(item, dict):
                for field in ['prompt']:
                    if field in item and item[field]:
                        prompt = str(item[field]).strip()
                        break
                if prompt is None:
                    print(f"Available keys in item {i}: {list(item.keys())}")
                    for key, value in item.items():
                        if value and isinstance(value, str) and len(value.strip()) > 0:
                            prompt = str(value).strip()
                            print(f"Using field '{key}' as prompt")
                            break
            elif isinstance(item, str):
                prompt = item.strip()
            else:
                prompt = str(item).strip()
            
            if prompt and len(prompt) > 0:
                data.append({
                    'prompt': prompt,
                    'original_item': item
                })
            else:
                print(f"Warning: Empty prompt for item {i}")
        
        print(f"Successfully processed {len(data)} valid prompts")
        
        if data:
            print("\nFirst few prompts for verification:")
            for i, item in enumerate(data[:3]):
                print(f"Sample {i+1}: {item['prompt'][:100]}...")
        
        return data
    
    except Exception as e:
        print(f"Error loading {data_file}: {e}")
        import traceback
        traceback.print_exc()
        return []


def main():
    parser = argparse.ArgumentParser(description="Test InsTagger checkpoint with data.pkl data using vLLM (Incremental Save)")
    parser.add_argument("--training_dir", type=str, default="./instagger-finetuned")
    parser.add_argument("--checkpoint_path", type=str, help="Specific checkpoint path")
    parser.add_argument("--data_path", type=str, required=True, 
                        help="Path to a .pkl data file for inference")
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--output_file", type=str, default="inference_results_with_raw.pkl")
    parser.add_argument("--save_raw_text", action="store_true", default=True)
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size for inference")
    parser.add_argument("--save_interval", type=int, default=500, help="Save interval for incremental saves")

    args = parser.parse_args()
    
    if args.checkpoint_path:
        model_path = args.checkpoint_path
    else:
        latest_checkpoint = find_latest_checkpoint(args.training_dir)
        if latest_checkpoint:
            model_path = latest_checkpoint
        else:
            final_model_path = os.path.join(args.training_dir, "final_model")
            if os.path.exists(final_model_path):
                model_path = final_model_path
            else:
                print("No checkpoints or final model found!")
                return
    
    print(f" Using model: {model_path}")
    print(f"Using data file: {args.data_path}")
    print(f"Batch size: {args.batch_size}")
    print(f"Save interval: {args.save_interval}")
    print(f"Output file: {args.output_file}")
    
    data = load_input_data(args.data_path, args.num_samples)
    if not data:
        print(f"No data loaded from {args.data_path}!")
        return
    
    try:
        results = run_inference_vllm(
            model_path, 
            data, 
            args.output_file, 
            args.save_raw_text, 
            save_interval=args.save_interval, 
            batch_size=args.batch_size
        )
        print(f"\nInference completed successfully!")
        print(f" Final results: {len(results)} samples")
        print(f" Results saved to: {args.output_file}")
        
    except Exception as e:
        print(f" Error during inference: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()