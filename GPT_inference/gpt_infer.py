#!/usr/bin/env python3
import os
import pickle
import json
import argparse
import time
import re
from openai import OpenAI

def create_prompt(instruction):
    """创建标签提取的提示"""
    return f"""You are a tagging system that provides useful tags for instruction intentions to distinguish instructions for a helpful AI assistant. Below is an instruction: [begin] {instruction} [end] Please provide coarse-grained tags, such as "Spelling and Grammar Check" and "Cosplay", to identify main intentions of above instruction. Your answer should be a list including titles of tags and a brief explanation of each tag. Your response have to strictly follow this JSON format: [{{"tag": str, "explanation": str}}]. Please response in English."""

def clean_json_response(response):
    """提取JSON内容"""
    
    response = response.strip()
    
    # 🔥 方法1: 直接JSON解析
    try:
        json_pattern = r'(\[.*?\]|\{.*?\})'
        matches = re.findall(json_pattern, response, re.DOTALL)
        if matches:
            longest_match = max(matches, key=len)
            # 验证是否为有效JSON
            json.loads(longest_match)
            return longest_match.strip()
    except:
        pass
    
    # 方法2: 正则提取（备用方案）
    try:
        tag_pattern = r'"tag":\s*"([^"]*)"'
        explanation_pattern = r'"explanation":\s*"([^"]*(?:[^"\\]|\\.)*)"\s*[,}]'
        
        tags = re.findall(tag_pattern, response)
        explanations = re.findall(explanation_pattern, response)
        
        if tags and explanations and len(tags) == len(explanations):
            # 重构为JSON字符串格式
            json_items = []
            for i in range(len(tags)):
                json_items.append(f'{{"tag": "{tags[i]}", "explanation": "{explanations[i]}"}}')
            return '[' + ', '.join(json_items) + ']'
    except:
        pass
    
    # 如果都失败，返回原始响应
    return response

def get_tags(client, instruction, model_name="gpt-3.5-turbo", max_tokens=512, temperature=0.6, top_p=0.9):
    """获取单个指令的标签"""
    prompt = create_prompt(instruction)
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        
        result = response.choices[0].message.content
        
        # 🔍 添加调试信息 - 保存原始响应
        print(f"\n=== DEBUG: Raw API Response ===")
        print(f"Model: {model_name}")
        print(f"Raw response length: {len(result)}")
        print(f"Raw response content:")
        print(f"'{result}'")
        
        # 🧹 清理markdown格式
        cleaned_result = clean_json_response(result)
        print(f"\n--- Cleaned JSON ---")
        print(f"'{cleaned_result}'")
        print(f"=== End Raw Response ===\n")
        
        # 简单解析JSON
        try:
            tags = json.loads(cleaned_result)
            print(f"✅ JSON解析成功: {tags}")
            return result, tags, True  # 🔥 返回原始响应
        except json.JSONDecodeError as e:
            print(f"❌ JSON解析失败: {e}")
            print(f"解析失败的内容: '{cleaned_result}'")
            return result, [{"tag": "Error", "explanation": "Failed to parse JSON"}], False
            
    except Exception as e:
        print(f"API调用失败: {e}")
        error_msg = str(e)
        return error_msg, [{"tag": "Error", "explanation": error_msg}], False

def load_data(data_path, limit=None):
    """加载数据并提取提示"""
    try:
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"Loaded {len(data)} samples from {data_path}")
        
        if limit and limit > 0:
            data = data[:limit]
            print(f"Limited to first {limit} samples")
        
        prompts = []
        for i, item in enumerate(data):
            prompt = ""
            
            if isinstance(item, dict):
                for field in ['prompt', 'instruction', 'query', 'text', 'content']:
                    if field in item and item[field]:
                        prompt = str(item[field]).strip()
                        break
                
                if not prompt:
                    for key, value in item.items():
                        if isinstance(value, str) and value.strip():
                            prompt = value.strip()
                            break
            elif isinstance(item, str):
                prompt = item.strip()
            else:
                prompt = str(item).strip()
            
            if prompt:
                prompts.append(prompt)
        
        print(f"Extracted {len(prompts)} valid prompts")
        return prompts
    except Exception as e:
        print(f"Error loading data: {e}")
        return []

def save_results(results, output_file):
    """保存结果"""
    try:
        with open(output_file, 'wb') as f:
            pickle.dump(results, f)
        print(f"Results saved to {output_file}")
        return True
    except Exception as e:
        print(f"Error saving results: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="ChatGPT Simple Inference")
    parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo", 
                        help="ChatGPT model name")
    parser.add_argument("--api_key", type=str, default=None,
                        help="OpenAI API key")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to input data file")
    parser.add_argument("--output_file", type=str, default="chatgpt_results.pkl",
                        help="Output file path")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of samples to process")
    parser.add_argument("--max_tokens", type=int, default=512,
                        help="Maximum tokens for generation")
    parser.add_argument("--temperature", type=float, default=0.6,
                        help="Temperature for generation")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p for generation")
    
    args = parser.parse_args()
    
    print(f"Configuration:")
    print(f"  Model: {args.model_name}")
    print(f"  Data: {args.data_path}")
    print(f"  Output: {args.output_file}")
    print(f"  Limit: {args.limit}")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Top-p: {args.top_p}")
    
    # 设置API key
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: Please provide API key via --api_key or OPENAI_API_KEY environment variable")
        return
    
    # 初始化客户端
    client = OpenAI(api_key=api_key)
    print("OpenAI client initialized successfully!")
    
    # 加载数据
    prompts = load_data(args.data_path, args.limit)
    if not prompts:
        print("No prompts loaded!")
        return
    
    # 处理提示
    print(f"Processing {len(prompts)} prompts...")
    results = []
    success_count = 0
    
    start_time = time.time()
    
    for i, prompt in enumerate(prompts):
        print(f"Processing {i+1}/{len(prompts)}")
        
        raw_response, tags, success = get_tags(  # 🔥 接收原始响应
            client, 
            prompt, 
            args.model_name, 
            args.max_tokens, 
            args.temperature, 
            args.top_p
        )
        
        if success:
            success_count += 1
        
        result = {
            "index": i,
            "prompt": prompt,
            "raw_response": raw_response,  # 🔥 保存完整的GPT原始响应
            "parsed_tags": tags,
            "success": success
        }
        results.append(result)
        
        # 每10条保存一次
        if (i + 1) % 10 == 0:
            save_results(results, args.output_file)
            print(f"Checkpoint saved: {i+1} results processed")
    
    # 最终保存
    save_results(results, args.output_file)
    
    # 统计信息
    inference_time = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"INFERENCE COMPLETED")
    print(f"{'='*60}")
    print(f"Total samples processed: {len(results)}")
    print(f"Successful inferences: {success_count}")
    print(f"Success rate: {success_count/len(results)*100:.1f}%")
    print(f"Total inference time: {inference_time:.2f} seconds")
    print(f"Average time per sample: {inference_time/len(results):.3f} seconds")
    print(f"Throughput: {len(results)/inference_time:.2f} samples/second")
    print(f"Results saved to: {args.output_file}")
    
    # 显示示例结果
    print(f"\nSample Results:")
    for i, result in enumerate(results[:3]):
        if result.get('success', False):
            print(f"\n--- Sample {i+1} ---")
            print(f"Prompt: {result['prompt'][:100]}...")
            tags = [tag['tag'] for tag in result['parsed_tags']]
            print(f"Tags: {tags}")

if __name__ == "__main__":
    main()