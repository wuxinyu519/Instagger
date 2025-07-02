#!/usr/bin/env python3
import os
import pickle
import json
import time
import re
import argparse
from typing import List, Dict, Any
from tqdm import tqdm

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

class GemmaVLLMInference:
    def __init__(self, model_name: str = "google/gemma-3-27b-it", 
                 tensor_parallel_size: int = 1, 
                 gpu_memory_utilization: float = 0.9):
        """Initialize Gemma model with vLLM"""
        print(f"Loading model: {model_name}")
        
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
            disable_log_stats=True,
            max_model_len=4096,
            # torch_dtype=torch.bfloat16,
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("Model loaded successfully!")
    
    def create_prompt(self, instruction: str) -> str:
        """Create prompt for instruction tagging"""
        return f"""You are a tagging system that provides useful tags for instruction intentions to distinguish instructions for a helpful AI assistant. Below is an instruction: [begin] {instruction} [end] Please provide coarse-grained tags, such as "Spelling and Grammar Check" and "Cosplay", to identify main intentions of above instruction. Your answer should be a list including titles of tags and a brief explanation of each tag. Your response have to strictly follow this JSON format: [{{"tag": str, "explanation": str}}]. Please response in English."""
        # return f"""You are a tagging system that provides useful tags for instruction intentions to distinguish privacy-related instructions for a helpful AI assistant. Below is an instruction: [begin] {instruction} [end] Please provide coarse-grained tags, such as "Spelling and Grammar Check" and "Cosplay", to identify main intentions of above instruction. Your answer should be a list including titles of tags and a brief explanation of each tag. Your response have to strictly follow this JSON format: [{{"tag": str, "explanation": str}}]. Please response in English."""
    
    def format_chat_input(self, instruction: str) -> str:
        """Format input for Gemma chat template"""
        prompt = self.create_prompt(instruction)
        
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        # messages = [{"role": "user", "content": prompt}]
        
        try:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            return f"<bos><start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
    
    def extract_json_from_response(self, response: str) -> List[Dict[str, str]]:
        """Extract JSON tags from model response"""
        try:
            response = response.strip()
            json_pattern = r'\[.*?\]'
            json_matches = re.findall(json_pattern, response, re.DOTALL)
            
            if json_matches:
                json_str = json_matches[-1]
                parsed_json = json.loads(json_str)
                
                if isinstance(parsed_json, list):
                    valid_tags = []
                    for item in parsed_json:
                        if isinstance(item, dict) and "tag" in item and "explanation" in item:
                            valid_tags.append({
                                "tag": str(item["tag"]).strip(),
                                "explanation": str(item["explanation"]).strip()
                            })
                    return valid_tags[:5] if valid_tags else self._fallback_parse(response)
            
            return self._fallback_parse(response)
        except Exception:
            return self._fallback_parse(response)
    
    def _fallback_parse(self, response: str) -> List[Dict[str, str]]:
        """Fallback parsing method"""
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
        except Exception:
            return [{"tag": "Error", "explanation": "Failed to parse response"}]
    
    def inference_batch_with_checkpoints(self, instructions: List[str], batch_size: int = 32, 
                                         output_file: str = None, checkpoint_interval: int = 10) -> List[Dict[str, Any]]:
        """Batch inference with incremental saving"""
        print(f"Processing {len(instructions)} instructions in batches of {batch_size}")
        
        # Try to load existing results
        all_results = []
        if output_file:
            all_results = load_existing_results(output_file)
        
        # Determine starting point
        start_idx = len(all_results)
        if start_idx > 0:
            print(f"Resuming from index {start_idx} (found {start_idx} existing results)")
            instructions = instructions[start_idx:]
        
        if not instructions:
            print("All instructions already processed!")
            return all_results
        
        formatted_inputs = [self.format_chat_input(inst) for inst in instructions]
        
        sampling_params = SamplingParams(
            temperature=0.6,
            top_p=0.9,
            max_tokens=512,
            stop=["<end_of_turn>", "</s>"],
        )
        
        batch_count = 0
        for i in tqdm(range(0, len(formatted_inputs), batch_size), desc="Processing batches"):
            batch_inputs = formatted_inputs[i:i + batch_size]
            batch_instructions = instructions[i:i + batch_size]
            
            try:
                outputs = self.llm.generate(batch_inputs, sampling_params)
                
                batch_results = []
                for j, output in enumerate(outputs):
                    instruction = batch_instructions[j]
                    formatted_input = batch_inputs[j]
                    generated_text = output.outputs[0].text.strip()
                    parsed_tags = self.extract_json_from_response(generated_text)
                    
                    result = {
                        "index": start_idx + i + j,
                        "instruction": instruction,
                        "formatted_input": formatted_input,
                        "raw_response": generated_text,
                        "parsed_tags": parsed_tags,
                        "success": len(parsed_tags) > 0 and parsed_tags[0]["tag"] != "Error"
                    }
                    batch_results.append(result)
                
                all_results.extend(batch_results)
                batch_count += 1
                
                # Incremental save every checkpoint_interval batches
                if output_file and batch_count % checkpoint_interval == 0:
                    save_results_incremental(all_results, output_file)
                    print(f"Checkpoint saved after batch {batch_count} ({len(all_results)} total results)")
                    
            except Exception as e:
                print(f"Error processing batch {i//batch_size + 1}: {e}")
                
                # Save partial results before handling error
                if output_file and all_results:
                    save_results_incremental(all_results, output_file)
                
                for j in range(len(batch_inputs)):
                    if i + j < len(instructions):
                        result = {
                            "index": start_idx + i + j,
                            "instruction": batch_instructions[j] if j < len(batch_instructions) else "",
                            "raw_response": f"Error: {str(e)}",
                            "parsed_tags": [{"tag": "Error", "explanation": f"Processing failed: {str(e)}"}],
                            "success": False
                        }
                        all_results.append(result)
        
        # Final save
        if output_file:
            save_results_incremental(all_results, output_file)
        
        return all_results

def load_data(data_path: str, limit: int = None) -> List[str]:
    """Load data and extract instructions"""
    try:
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"Loaded {len(data)} samples from {data_path}")
        
        if limit and limit > 0:
            data = data[:limit]
            print(f"Limited to first {limit} samples")
        
        instructions = []
        for i, item in enumerate(data):
            instruction = ""
            
            if isinstance(item, dict):
                for field in ['prompt', 'instruction', 'query', 'text', 'content']:
                    if field in item and item[field]:
                        instruction = str(item[field]).strip()
                        break
                
                if not instruction:
                    for key, value in item.items():
                        if isinstance(value, str) and value.strip():
                            instruction = value.strip()
                            break
            elif isinstance(item, str):
                instruction = item.strip()
            else:
                instruction = str(item).strip()
            
            if instruction:
                instructions.append(instruction)
        
        print(f"Extracted {len(instructions)} valid instructions")
        return instructions
    except Exception as e:
        print(f"Error loading data: {e}")
        return []

def save_results_incremental(results: List[Dict], output_file: str):
    """Save results with incremental checkpointing"""
    try:
        # Save main results
        with open(output_file, 'wb') as f:
            pickle.dump(results, f)
        
        # Create checkpoint file name
        checkpoint_file = output_file.replace('.pkl', '_checkpoint.pkl')
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"Results saved to {output_file} (checkpoint: {checkpoint_file})")
        return True
    except Exception as e:
        print(f"Error saving results: {e}")
        return False

def load_existing_results(output_file: str) -> List[Dict]:
    """Load existing results for resume functionality"""
    checkpoint_file = output_file.replace('.pkl', '_checkpoint.pkl')
    
    # Try to load from checkpoint first
    for file_path in [checkpoint_file, output_file]:
        if os.path.exists(file_path):
            try:
                with open(file_path, 'rb') as f:
                    results = pickle.load(f)
                print(f"Loaded {len(results)} existing results from {file_path}")
                return results
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
    
    return []

def main():
    parser = argparse.ArgumentParser(description="Simplified Gemma vLLM Inference")
    parser.add_argument("--model_name", type=str, default="google/gemma-2-27b-it")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="gemma_results.pkl")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--checkpoint_interval", type=int, default=10,
                        help="Save checkpoint every N batches")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing checkpoint")
    
    args = parser.parse_args()
    
    print(f"Configuration:")
    print(f"  Model: {args.model_name}")
    print(f"  Data: {args.data_path}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Tensor parallel size: {args.tensor_parallel_size}")
    print(f"  GPU memory utilization: {args.gpu_memory_utilization}")
    print(f"  Checkpoint interval: {args.checkpoint_interval}")
    print(f"  Resume mode: {args.resume}")
    print(f"  Limit: {args.limit}")
    
    # Check if resuming and output file exists
    if args.resume and os.path.exists(args.output_file):
        print(f"Resume mode enabled - will continue from existing results")
    elif args.resume:
        print(f"Resume mode enabled but no existing file found - starting fresh")
    
    # Load data
    instructions = load_data(args.data_path, args.limit)
    if not instructions:
        print("No instructions loaded!")
        return
    
    # Initialize model
    start_time = time.time()
    inference_engine = GemmaVLLMInference(
        model_name=args.model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization
    )
    load_time = time.time() - start_time
    print(f"Model loading time: {load_time:.2f} seconds")
    
    # Run inference with checkpoints
    start_time = time.time()
    results = inference_engine.inference_batch_with_checkpoints(
        instructions=instructions,
        batch_size=args.batch_size,
        output_file=args.output_file,
        checkpoint_interval=args.checkpoint_interval
    )
    inference_time = time.time() - start_time
    
    # Final save
    save_results_incremental(results, args.output_file)
    
    # Print statistics
    if results:
        success_count = sum(1 for r in results if r.get('success', False))
        
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
        
        # Show sample results
        print(f"\nSample Results:")
        for i, result in enumerate(results[:3]):
            if result.get('success', False):
                print(f"\n--- Sample {i+1} ---")
                print(f"Instruction: {result['instruction'][:100]}...")
                tags = [tag['tag'] for tag in result['parsed_tags']]
                print(f"Tags: {tags}")

if __name__ == "__main__":
    main()