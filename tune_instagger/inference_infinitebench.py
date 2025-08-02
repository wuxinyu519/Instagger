#!/usr/bin/env python3
import os
import pickle
import json
import re
import numpy as np
from tqdm import tqdm
import argparse
from transformers import AutoTokenizer
from vllm import SamplingParams, LLM
import time
import glob

def is_huggingface_model_id(model_path: str) -> bool:
    """
    判断是否为Hugging Face模型ID
    如果路径存在于本地文件系统，则认为是本地路径；否则认为是HF模型ID
    """
    return not os.path.exists(model_path)

def get_model_name_from_path(model_path: str) -> str:
    """
    从模型路径提取模型名称作为目录名
    """
    if is_huggingface_model_id(model_path):
        # HF模型ID，如 "microsoft/DialoGPT-medium" -> "microsoft_DialoGPT-medium"
        return model_path.replace('/', '_').replace('\\', '_')
    else:
        # 本地路径，取最后一个目录名，如 "/path/to/final_model" -> "final_model"
        return os.path.basename(os.path.normpath(model_path))

def truncate_context(context: str, tokenizer, max_tokens: int = 600) -> str:
    """
    更高效地截断context，只编码必要部分，避免重复和全量encode。
    保留前300和后300 tokens，中间不重叠。
    """
    max_each = max_tokens // 2  # 默认各保留300个
    rough_char_per_token = 4  # 估计每个token约4字符，可根据语言类型微调

    head_char_len = max_each * rough_char_per_token
    tail_char_len = max_each * rough_char_per_token

    total_len = len(context)

    # 如果字符长度太短，说明本来就不需要截断
    if total_len <= head_char_len + tail_char_len:
        return context

    # 分别截取前后段，确保中间不重叠
    front_chunk = context[:head_char_len]
    back_chunk = context[-tail_char_len:] if total_len > head_char_len + tail_char_len else context[head_char_len:]

    # 分别tokenize
    front_tokens = tokenizer.encode(front_chunk, add_special_tokens=False)[:max_each]
    back_tokens = tokenizer.encode(back_chunk, add_special_tokens=False)[-max_each:]

    # decode
    front_text = tokenizer.decode(front_tokens, skip_special_tokens=True)
    back_text = tokenizer.decode(back_tokens, skip_special_tokens=True)

    # 组合（中间用省略标记）
    return front_text + "\n\n[...]\n\n" + back_text

def load_instagger_vllm(model_path, tensor_parallel_size=1):
    """使用 vLLM 加载模型和 tokenizer - 本地模型和HF模型"""
    
    is_hf_model = is_huggingface_model_id(model_path)
    
    if is_hf_model:
        print(f"Loading Hugging Face model: {model_path}")
    else:
        print(f"Loading local model from: {model_path}")
        # 检查本地模型路径是否存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Local model path not found: {model_path}")
        
        # 检查是否包含必要的模型文件
        required_files = ['config.json']
        for file in required_files:
            file_path = os.path.join(model_path, file)
            if not os.path.exists(file_path):
                print(f"Warning: {file} not found in {model_path}")
    
    # 配置vLLM参数
    llm_kwargs = {
        "model": model_path,
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": 0.8,
        "disable_log_stats": True,
        "trust_remote_code": True,
        "enforce_eager": True,
        "max_model_len": 2048,
    }
    
    # 只对本地模型设置download_dir=None
    if not is_hf_model:
        llm_kwargs["download_dir"] = None
    
    # 多GPU配置
    if tensor_parallel_size > 1:
        llm_kwargs["distributed_executor_backend"] = "mp"
        print(f"Using multi-GPU mode with {tensor_parallel_size} GPUs")
    else:
        print("Using single GPU mode")
    
    try:
        print("Initializing vLLM engine...")
        llm = LLM(**llm_kwargs)
        print("vLLM engine loaded successfully")
    except Exception as e:
        print(f"Failed to load with tensor_parallel_size={tensor_parallel_size}: {e}")
        print("Trying single GPU mode...")
        # 回退到单GPU模式
        llm_kwargs["tensor_parallel_size"] = 1
        if "distributed_executor_backend" in llm_kwargs:
            del llm_kwargs["distributed_executor_backend"]
        try:
            llm = LLM(**llm_kwargs)
            print("Successfully loaded in single GPU mode")
        except Exception as e2:
            print(f"Failed to load even in single GPU mode: {e2}")
            raise e2

    print("Loading tokenizer...")
    try:
        tokenizer_kwargs = {
            "use_fast": False,
            "trust_remote_code": True,
        }
        
        # 只对本地模型强制使用本地文件
        if not is_hf_model:
            tokenizer_kwargs["local_files_only"] = True
        
        tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_kwargs)
        print("Tokenizer loaded successfully")
        
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        if not is_hf_model:
            print("Trying without local_files_only...")
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                use_fast=False,
                trust_remote_code=True,
            )
        else:
            raise e
    
    # 设置pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Set pad_token to eos_token")

    # 打印EOS token信息
    print(f"EOS token: {tokenizer.eos_token}")
    print(f"EOS token ID: {tokenizer.eos_token_id}")

    return llm, tokenizer

def extract_tags_with_explanations(tags_text):
    """提取标签和解释 - 改进的JSON解析"""
    try:
        # 寻找JSON数组
        json_pattern = r'\[.*?\]'
        json_matches = re.findall(json_pattern, tags_text, re.DOTALL)
        
        if json_matches:
            json_str = json_matches[-1]
            try:
                parsed_json = json.loads(json_str)
                if isinstance(parsed_json, list):
                    valid_tags = []
                    for item in parsed_json:
                        if isinstance(item, dict) and "tag" in item and "explanation" in item:
                            valid_tags.append({
                                "tag": str(item["tag"]).strip(),
                                "explanation": str(item["explanation"]).strip()
                            })
                    return valid_tags[:5]
            except json.JSONDecodeError:
                pass
        
        # 寻找单个JSON对象
        single_json_pattern = r'\{[^{}]*"tag"[^{}]*"explanation"[^{}]*\}'
        single_matches = re.findall(single_json_pattern, tags_text)
        
        if single_matches:
            valid_tags = []
            for match in single_matches:
                try:
                    item = json.loads(match)
                    if "tag" in item and "explanation" in item:
                        valid_tags.append({
                            "tag": str(item["tag"]).strip(),
                            "explanation": str(item["explanation"]).strip()
                        })
                except:
                    continue
            return valid_tags[:5]
        
        # 回退方案：简单解析
        return _fallback_parse(tags_text)
        
    except Exception as e:
        print(f"Error extracting tags with explanations: {e}")
        return []

def _fallback_parse(response: str):
    """回退解析方法"""
    try:
        tags = []
        lines = response.split('\n')
        current_tag = None
        current_explanation = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if 'tag' in line.lower() and ':' in line:
                if current_tag and current_explanation:
                    tags.append({"tag": current_tag, "explanation": current_explanation})
                
                parts = line.split(':', 1)
                if len(parts) == 2:
                    current_tag = parts[1].strip().replace('"', '').replace("'", "")
                    current_explanation = None
            
            elif 'explanation' in line.lower() and ':' in line:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    current_explanation = parts[1].strip().replace('"', '').replace("'", "")
            
            elif current_tag and not current_explanation and line:
                current_explanation = line.replace('"', '').replace("'", "")
        
        if current_tag and current_explanation:
            tags.append({"tag": current_tag, "explanation": current_explanation})
        
        return tags[:5] if tags else [{"tag": "General", "explanation": "Unable to parse specific tags"}]
    except:
        return [{"tag": "Error", "explanation": "Failed to parse response"}]

def extract_tags(tags_text):
    tags_with_explanations = extract_tags_with_explanations(tags_text)
    return [item["tag"] for item in tags_with_explanations]

def append_results_to_file(output_file, new_results):
    """追加结果到文件"""
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
                       save_raw_text=True, save_interval=64, batch_size=32,
                       tensor_parallel_size=1):
    
    print("Loading model...")
    llm, tokenizer = load_instagger_vllm(model_path, tensor_parallel_size)
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
        
        # 构建输入文本，对inference_context进行截断处理
        input_texts = []
        for item in batch:
            # 使用inference_context进行推理
            inference_text = item.get('inference_context', item.get('context', ''))
            truncated_context = truncate_context(inference_text, tokenizer)
            
            # 构建prompt
            prompt_text = f"You are a helpful assistant. Please identify tags of user intentions in the following user query and provide an explanation for each tag. Please respond in the JSON format {{\"tag\": str, \"explanation\": str}}. Query: {truncated_context} Assistant:"
            input_texts.append(prompt_text)

        eos_tokens = []
        if tokenizer.eos_token:
            eos_tokens.append(tokenizer.eos_token)

        # 添加常见的停止标记作为备用
        stop_tokens = eos_tokens + ["</s>", "<end_of_turn>"]

        sampling_params = SamplingParams(
            temperature=0.9,
            top_p=0.6,
            max_tokens=512,
            stop=stop_tokens,
        )

        try:
            responses = llm.generate(input_texts, sampling_params)
        except Exception as e:
            print(f"Error during vLLM batch generation: {e}")
            responses = [None] * len(batch)

        # 处理批次的结果
        current_batch_results = []
        for i, item in enumerate(batch):
            idx = start + i
            try:
                if responses[i] is not None:
                    raw_text = responses[i].outputs[0].text
            
                    prefix = input_texts[i]
                    if raw_text.startswith(prefix):
                        generated_part = raw_text[len(prefix):].strip()
                    else:
                        generated_part = raw_text

                    # 提取标签和解释
                    parsed_tags_with_explanations = extract_tags_with_explanations(generated_part)
                else:
                    raw_text = "Generation failed"
                    parsed_tags_with_explanations = []

                #保持原始数据 + 添加三个新字段
                result = {
                    **item, 
                    'truncated_input': truncate_context(
                        item.get('inference_context', item.get('context', '')), tokenizer
                    ),
                    'generated_tags': parsed_tags_with_explanations,
                    'raw_response': raw_text if save_raw_text else None,
                }

                current_batch_results.append(result)

            except Exception as e:
                print(f"Error processing sample {idx+1}: {e}")
                result = {
                    **item,  # 保持原始数据
                    'truncated_input': '',
                    'generated_tags': [],
                    'raw_response': f"Error: {str(e)}" if save_raw_text else None,
                }
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

        success_count = sum(1 for r in final_results if r.get('generated_tags', []))
        print(f"Successfully processed: {success_count}/{len(final_results)} samples")
        
        new_results_start = max(0, len(final_results) - total_processed)
        preview_results = final_results[new_results_start:new_results_start + 3]
        
        for i, result in enumerate(preview_results):
            print(f"\n--- Sample {new_results_start + i + 1} ---")
            generated_tags = result.get('generated_tags', [])
            print(f"Generated Tags: {[tag_info['tag'] for tag_info in generated_tags]}")
            if generated_tags:
                print(f"First 2 Tags with Explanations: {generated_tags[:2]}")
            print("-" * 60)

        print("="*80)
        
        return final_results
        
    except Exception as e:
        print(f"Warning: Could not verify final results: {e}")
        return []

def load_single_json_file(json_file: str, num_samples: int = None):
    """从单个.json或.jsonl文件加载数据"""
    try:
        file_data = []
        
        with open(json_file, 'r', encoding='utf-8') as f:
            if json_file.endswith('.jsonl'):
                
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                        processed_item = process_item(item, json_file, line_num)
                        if processed_item:
                            file_data.append(processed_item)
                    except json.JSONDecodeError as e:
                        print(f"Warning: JSON decode error at line {line_num} in {json_file}: {e}")
                        continue
            else:
                # JSON格式：整个文件是一个JSON数组或对象
                try:
                    data = json.load(f)
                    if isinstance(data, list):
                        for idx, item in enumerate(data, 1):
                            processed_item = process_item(item, json_file, idx)
                            if processed_item:
                                file_data.append(processed_item)
                    elif isinstance(data, dict):
                        processed_item = process_item(data, json_file, 1)
                        if processed_item:
                            file_data.append(processed_item)
                except json.JSONDecodeError as e:
                    print(f"Error: Invalid JSON in {json_file}: {e}")
                    return []
        
        # 限制样本数量
        if num_samples is not None and num_samples > 0:
            file_data = file_data[:num_samples]
        
        return file_data
    except Exception as e:
        print(f"Error loading {json_file}: {e}")
        return []

def process_item(item, file_name, line_num):
    """处理单个数据项"""
    if not isinstance(item, dict):
        print(f"Warning: Item at line {line_num} in {file_name} is not a dict")
        return None
    
    # 确保有input字段
    if 'input' not in item:
        print(f"Warning: Line {line_num} in {file_name} missing 'input' field")
        return None
    
    # 获取context和input字段，创建用于推理的context
    context_content = item.get('context', '')
    input_content = item['input']
    
    # 连接context和input字段，添加为新字段用于推理
    if context_content:
        item['inference_context'] = f"{input_content}\n\n{context_content}"
    else:
        item['inference_context'] = input_content
    
    return item

def find_json_files(data_dir: str):
    """查找所有.json和.jsonl文件"""
    if not os.path.exists(data_dir):
        print(f"Error: {data_dir} directory not found!")
        return []
    
    # 查找所有.json和.jsonl文件 - 递归搜索
    all_files = glob.glob(os.path.join(data_dir, "**", "*.json*"), recursive=True)
    
    if not all_files:
        print(f"No .json or .jsonl files found in {data_dir}")
        return []
    
    print(f"Found {len(all_files)} files:")
    for file in all_files:
        rel_path = os.path.relpath(file, data_dir)
        print(f"  - {rel_path}")
    
    return all_files

def process_single_file(model_path: str, json_file: str, output_dir: str, args):
    """处理单个JSON文件"""
    # 计算相对于data_dir的路径
    rel_path = os.path.relpath(json_file, args.data_dir)
    
    # 构建输出路径，保持目录结构
    rel_dir = os.path.dirname(rel_path)
    file_basename = os.path.splitext(os.path.basename(rel_path))[0]
    
    # 创建对应的输出子目录
    output_subdir = os.path.join(output_dir, rel_dir) if rel_dir else output_dir
    os.makedirs(output_subdir, exist_ok=True)
    
    output_file = os.path.join(output_subdir, f"{file_basename}_results.pkl")
    
    print(f"\n{'='*80}")
    print(f"Processing file: {json_file}")
    print(f"Output file: {output_file}")
    print(f"{'='*80}")
    
    # 加载单个文件的数据
    data = load_single_json_file(json_file, args.num_samples)
    if not data:
        print(f"No data loaded from {json_file}!")
        return None
    
    print(f"Loaded {len(data)} samples from {os.path.basename(json_file)}")
    
    # 运行推理
    try:
        results = run_inference_vllm(
            model_path, 
            data, 
            output_file,
            args.save_raw_text, 
            save_interval=args.save_interval, 
            batch_size=args.batch_size,
            tensor_parallel_size=args.tensor_parallel_size
        )
        print(f"\n✓ Successfully processed {json_file}")
        print(f"  Results: {len(results)} samples")
        print(f"  Output saved to: {output_file}")
        return output_file
        
    except Exception as e:
        print(f"✗ Error processing {json_file}: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    parser = argparse.ArgumentParser(description="Inference with local model or HF model - generate tags with explanations for JSON/JSONL data")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Local model directory path or Hugging Face model ID")
    parser.add_argument("--data_dir", type=str, default="data", 
                        help="Directory containing .json/.jsonl files (default: 'data')")
    parser.add_argument("--output_prefix", type=str, default="results",
                        help="Prefix for output directory name (default: 'results')")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Limit number of samples per file (default: process all)")
    parser.add_argument("--save_raw_text", action="store_true", default=True)
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for inference")
    parser.add_argument("--save_interval", type=int, default=100, help="Save interval for incremental saves")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Number of GPUs for tensor parallel")

    args = parser.parse_args()
    
    # 判断是本地模型还是HF模型
    model_path = args.checkpoint_path
    is_hf_model = is_huggingface_model_id(model_path)

    if is_hf_model:
        print(f"Using Hugging Face model: {model_path}")
    else:
        print(f"Using local model: {model_path}")
        # 只对本地模型验证路径存在性
        if not os.path.exists(model_path):
            print(f"Error: Local model path does not exist: {model_path}")
            return
    
    # 创建以模型名命名的输出目录
    model_name = get_model_name_from_path(model_path)
    output_dir = f"{args.output_prefix}_{model_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Model path: {model_path}")
    print(f"Output directory: {output_dir}")
    print(f"Using data directory: {args.data_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Tensor parallel size: {args.tensor_parallel_size}")
    print(f"Save interval: {args.save_interval}")
    
    # 检查数据目录
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory does not exist: {args.data_dir}")
        return
    
    # 查找所有JSON文件
    json_files = find_json_files(args.data_dir)
    if not json_files:
        print(f"No JSON files found in {args.data_dir}!")
        return
    
    # 处理每个文件
    processed_files = []
    failed_files = []
    
    print(f"\nStarting to process {len(json_files)} files...")
    
    for i, json_file in enumerate(json_files, 1):
        print(f"\nProcessing file {i}/{len(json_files)}: {os.path.relpath(json_file, args.data_dir)}")
        
        result = process_single_file(model_path, json_file, output_dir, args)
        
        if result:
            processed_files.append(result)
        else:
            failed_files.append(json_file)
    
    # 输出总结
    print(f"\n{'='*80}")
    print("PROCESSING SUMMARY")
    print(f"{'='*80}")
    print(f"Total files: {len(json_files)}")
    print(f"Successfully processed: {len(processed_files)}")
    print(f"Failed: {len(failed_files)}")
    
    if processed_files:
        print(f"\nSuccessfully processed files:")
        for output_file in processed_files:
            print(f"  - {output_file}")
    
    if failed_files:
        print(f"\nFailed files:")
        for failed_file in failed_files:
            print(f"  - {failed_file}")
    
    print(f"\n📂 All results saved in directory: {output_dir}")
    print("✨ Processing completed!")

if __name__ == "__main__":
    main()