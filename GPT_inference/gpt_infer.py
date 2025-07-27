#!/usr/bin/env python3
import os
import pickle
import json
import argparse
import time
from openai import OpenAI

def create_prompt(instruction):
    """创建标签提取的提示"""
    return f"""You are a tagging system that provides useful tags for instruction intentions to distinguish instructions for a helpful AI assistant. Below is an privacy-related instruction: [begin] {instruction} [end] Please provide coarse-grained tags, such as "Spelling and Grammar Check" and "Cosplay", to identify main intentions of above instruction. Your answer should be a list including titles of tags and a brief explanation of each tag. Your response have to strictly follow this JSON format: [{{"tag": str, "explanation": str}}]. Please response in English."""

def clean_json_response(response):
    """智能清理GPT响应，提取JSON内容"""
    import re
    
    response = response.strip()
    
    # 方法1: 使用正则表达式提取JSON数组或对象
    # 匹配 [...] 或 {...}
    json_pattern = r'(\[.*?\]|\{.*?\})'
    matches = re.findall(json_pattern, response, re.DOTALL)
    
    if matches:
        # 返回最长的匹配（通常是完整的JSON）
        longest_match = max(matches, key=len)
        return longest_match.strip()
    
    # 方法2: 移除常见的非JSON前缀和后缀
    # 移除可能的markdown代码块标记
    lines = response.split('\n')
    start_idx = 0
    end_idx = len(lines)
    
    # 找到第一个包含 [ 或 { 的行
    for i, line in enumerate(lines):
        if '[' in line or '{' in line:
            start_idx = i
            break
    
    # 从后往前找到最后一个包含 ] 或 } 的行
    for i in range(len(lines) - 1, -1, -1):
        if ']' in lines[i] or '}' in lines[i]:
            end_idx = i + 1
            break
    
    # 重新组合有效的JSON部分
    cleaned_lines = lines[start_idx:end_idx]
    cleaned_response = '\n'.join(cleaned_lines).strip()
    
    # 方法3: 如果上述方法都失败，尝试简单的字符串清理
    if not cleaned_response:
        # 移除常见的非JSON字符
        prefixes_to_remove = ['```json', '```', 'json', 'JSON:', 'Response:', 'Here is the JSON:']
        suffixes_to_remove = ['```', '```json']
        
        for prefix in prefixes_to_remove:
            if response.lower().startswith(prefix.lower()):
                response = response[len(prefix):].strip()
        
        for suffix in suffixes_to_remove:
            if response.lower().endswith(suffix.lower()):
                response = response[:-len(suffix)].strip()
        
        cleaned_response = response
    
    return cleaned_response

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
            return tags, True
        except json.JSONDecodeError as e:
            print(f"❌ JSON解析失败: {e}")
            print(f"解析失败的内容: '{cleaned_result}'")
            return [{"tag": "Error", "explanation": "Failed to parse JSON"}], False
            
    except Exception as e:
        print(f"API调用失败: {e}")
        return [{"tag": "Error", "explanation": str(e)}], False

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
        
        tags, success = get_tags(
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
            "raw_response": json.dumps(tags),
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