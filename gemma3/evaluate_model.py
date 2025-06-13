import os
import shutil
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from dataset_loader import load_dataset
from tqdm import tqdm
from collections import Counter
def merge_lora_model():
    """merge lora"""
    print("Merging LoRA adapter with base model...")
    
    base_model_name = 'google/gemma-3-1b-it'
    lora_checkpoint_path = './outputs/router_selector_1b/checkpoint-29242'
    merged_model_path = './outputs/router_selector_1b_merged'
    
    # check LoRA checkpoint
    if not os.path.exists(lora_checkpoint_path):
        raise FileNotFoundError(f"LoRA checkpoint not found at {lora_checkpoint_path}")
    
    # remove dir
    if os.path.exists(merged_model_path):
        
        print(f"Removing existing merged model directory...")
        shutil.rmtree(merged_model_path)
    
    try:
        
        print("Loading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
     
        print("Loading LoRA adapter...")
        peft_model = PeftModel.from_pretrained(base_model, lora_checkpoint_path)
        
      
        print("Merging weights...")
        merged_model = peft_model.merge_and_unload()
        
        # save
        os.makedirs(merged_model_path, exist_ok=True)
        
      
        print(f"Saving merged model to {merged_model_path}...")
        merged_model.save_pretrained(
            merged_model_path,
            safe_serialization=True,  
            max_shard_size="5GB"
        )
        
        # save tokenizer
        print("Saving tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        tokenizer.save_pretrained(merged_model_path)
        
        
        required_files = ['config.json', 'tokenizer.json', 'tokenizer_config.json']
        for file in required_files:
            file_path = os.path.join(merged_model_path, file)
            if os.path.exists(file_path):
                print(f"{file} saved successfully")
            else:
                print(f"{file} not found")
        
       
        weight_files = [f for f in os.listdir(merged_model_path) if f.endswith(('.bin', '.safetensors'))]
        if weight_files:
            print(f"Model weights saved: {weight_files}")
        else:
            print("No model weight files found")
        
        print("Model merging completed successfully!")
        
        
        del base_model, peft_model, merged_model
        torch.cuda.empty_cache()
        
        return merged_model_path
        
    except Exception as e:
        print(f"Error during model merging: {e}")
        raise

def evaluate_dataset(llm, tokenizer, sampling_params, data, dataset_name):
   
    print(f"\nEvaluating {dataset_name} dataset...")
    print(f"Dataset size: {len(data)} samples")
    
    def preprocess_for_inference(example):
        CANDIDATE_MODELS = [
            "WizardLM/WizardLM-13B-V1.2",
            "claude-instant-v1",
            "claude-v1", 
            "claude-v2",
            "gpt-3.5-turbo-1106",
            "gpt-4-1106-preview",
            "meta/code-llama-instruct-34b-chat",
            "meta/llama-2-70b-chat",
            "mistralai/mistral-7b-chat",
            "mistralai/mixtral-8x7b-chat",
            "no_model_correct",
            "zero-one-ai/Yi-34B-Chat"
        ]
        
        candidates_str = "\n".join([f"- {model}" for model in CANDIDATE_MODELS])
        
        return f"""Request: {example['prompt']}

                Task: {example['eval_name']}
                Tags: {', '.join(example['tags'])}

                Available models:
                {candidates_str}

                Select the best model to solve:"""
    
    
    prompts = []
    expected_outputs = []
    
    for example in data:
        full_prompt = preprocess_for_inference(example)
        prompts.append(full_prompt)
        expected_outputs.append(example['oracle_model_to_route_to'])
    
    
    print(f"Generating responses for {dataset_name}...")
    outputs = llm.generate(prompts, sampling_params)
    
    # acc
    correct = 0
    total = len(data)
    detailed_results = []
    progress_bar = tqdm(zip(outputs, expected_outputs), total=total, desc=f"{dataset_name} Accuracy: 0.00%")

    for i, (output, expected) in enumerate(zip(outputs, expected_outputs)):
        generated = output.outputs[0].text.strip()
        
        # check if match
        # is_correct = expected.strip().lower() == generated.strip().lower()
        is_correct = expected.strip().lower() in generated.strip().lower().split()

        if is_correct:
            correct += 1
        
        detailed_results.append({
            'expected': expected,
            'generated': generated,
            'correct': is_correct
        })

        current_acc = correct / (i + 1)
        progress_bar.set_description(f"{dataset_name} Accuracy: {current_acc:.2%}")

        
        # preview
        if dataset_name.lower() == "train" and i < 3:
            original_tokens = tokenizer(prompts[i])["input_ids"]
            is_truncated = len(original_tokens) > 2048
            
            print(f"\n{'='*60}")
            print(f"--- {dataset_name} Example {i+1} ---")
            print(f"\n input prompt:")
            print(f"{prompts[i]}")
            print(f"\nPrompt record:")
            print(f"- len: {len(prompts[i])}")
            print(f"- num of token: {len(original_tokens)}")
            print(f"- if truncated: {'yes' if is_truncated else 'no'}")
            
            print(f"\n predict:")
            print(f"- Expected: '{expected}'")
            print(f"- Generated: '{generated}'")
            print(f"- Match: {is_correct}")
            print(f"{'='*60}")
        


    final_accuracy = correct / total
    
    
    truncated_count = 0
    total_tokens = 0
    for prompt in prompts:
        tokens = tokenizer(prompt)["input_ids"]
        total_tokens += len(tokens)
        if len(tokens) > 2048:
            truncated_count += 1
    
    
    
    predictions = [r['generated'] for r in detailed_results]
    expected_list = [r['expected'] for r in detailed_results]
    
    pred_distribution = Counter(predictions)
    expected_distribution = Counter(expected_list)
    
    # 显示失败案例
    failed_cases = [r for r in detailed_results if not r['correct']]
    
    return {
        'dataset_name': dataset_name,
        'total_samples': total,
        'correct_predictions': correct,
        'accuracy': final_accuracy,
        'truncated_count': truncated_count,
        'truncated_ratio': truncated_count / total,
        'avg_token_length': total_tokens / total,
        'prediction_distribution': dict(pred_distribution),
        'expected_distribution': dict(expected_distribution),
        'failed_cases': failed_cases[:5]  # 前5个失败案例
    }

def evaluate_with_vllm(merged_model_path):

    
    print("Loading merged model with vLLM...")
    llm = LLM(
        model=merged_model_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.8,
        max_model_len=2048,
        dtype="float16"
    )
    
    # use tokenizer
    tokenizer = AutoTokenizer.from_pretrained(merged_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # sampling
    sampling_params = SamplingParams(
        max_tokens=20,
        temperature=0.0,
        top_p=1.0,
        stop=["\n", "</s>", tokenizer.eos_token],
        skip_special_tokens=True
    )
    
    # load dataset
    print("Loading datasets...")
    train_data = load_dataset('data/train_samples.pkl')
    val_data = load_dataset('data/val_samples.pkl')
    

    train_results = evaluate_dataset(llm, tokenizer, sampling_params, train_data, "Train")
    

    val_results = evaluate_dataset(llm, tokenizer, sampling_params, val_data, "Validation")
    
    # 打印结果总结
    print(f"\n{'='*80}")
    print(f"EVALUATION RESULTS SUMMARY")
    print(f"{'='*80}")
    
    print(f"\nTRAIN SET RESULTS:")
    print(f"Total samples: {train_results['total_samples']}")
    print(f"Correct predictions: {train_results['correct_predictions']}")
    print(f"Accuracy: {train_results['accuracy']:.2%}")
    print(f"Truncated samples: {train_results['truncated_count']}/{train_results['total_samples']} ({train_results['truncated_ratio']:.1%})")
    print(f"Average token length: {train_results['avg_token_length']:.1f}")
    
    print(f"\nVALIDATION SET RESULTS:")
    print(f"Total samples: {val_results['total_samples']}")
    print(f"Correct predictions: {val_results['correct_predictions']}")
    print(f"Accuracy: {val_results['accuracy']:.2%}")
    print(f"Truncated samples: {val_results['truncated_count']}/{val_results['total_samples']} ({val_results['truncated_ratio']:.1%})")
    print(f"Average token length: {val_results['avg_token_length']:.1f}")
    
    print(f"\nCOMPARISON:")
    acc_diff = train_results['accuracy'] - val_results['accuracy']
    
    print(f"Accuracy difference(train-val) is {acc_diff:.1%}")
    
    print(f"\n📊 PREDICTION DISTRIBUTION COMPARISON:")
    all_predictions = set(train_results['prediction_distribution'].keys()) | set(val_results['prediction_distribution'].keys())
    
    print(f"{'Model':<30} {'Train Count':<12} {'Val Count':<10} {'Train %':<8} {'Val %':<8}")
    print(f"{'-'*70}")
    for pred in sorted(all_predictions):
        train_count = train_results['prediction_distribution'].get(pred, 0)
        val_count = val_results['prediction_distribution'].get(pred, 0)
        train_pct = train_count / train_results['total_samples'] * 100
        val_pct = val_count / val_results['total_samples'] * 100
        print(f"{pred:<30} {train_count:<12} {val_count:<10} {train_pct:<7.1f}% {val_pct:<7.1f}%")
    
    # error samples
    if train_results['failed_cases']:
        print(f"\nTRAIN SET FAILED CASES (sample):")
        for i, case in enumerate(train_results['failed_cases'][:3]):
            print(f"  {i+1}. Expected: '{case['expected']}' | Generated: '{case['generated']}'")
    
    if val_results['failed_cases']:
        print(f"\nVALIDATION SET FAILED CASES (sample):")
        for i, case in enumerate(val_results['failed_cases'][:3]):
            print(f"  {i+1}. Expected: '{case['expected']}' | Generated: '{case['generated']}'")
    
    print(f"\n{'='*80}")

        # save
    with open("evaluation_results.txt", "w", encoding="utf-8") as f:
        f.write("EVALUATION RESULTS SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        f.write("TRAIN SET RESULTS:\n")
        f.write(f"Total samples: {train_results['total_samples']}\n")
        f.write(f"Correct predictions: {train_results['correct_predictions']}\n")
        f.write(f"Accuracy: {train_results['accuracy']:.2%}\n")
        f.write(f"Truncated samples: {train_results['truncated_count']}/{train_results['total_samples']} ({train_results['truncated_ratio']:.1%})\n")
        f.write(f"Average token length: {train_results['avg_token_length']:.1f}\n\n")
        
        f.write("VALIDATION SET RESULTS:\n")
        f.write(f"Total samples: {val_results['total_samples']}\n")
        f.write(f"Correct predictions: {val_results['correct_predictions']}\n")
        f.write(f"Accuracy: {val_results['accuracy']:.2%}\n")
        f.write(f"Truncated samples: {val_results['truncated_count']}/{val_results['total_samples']} ({val_results['truncated_ratio']:.1%})\n")
        f.write(f"Average token length: {val_results['avg_token_length']:.1f}\n\n")

        f.write("PREDICTION DISTRIBUTION COMPARISON:\n")
        f.write(f"{'Model':<30} {'Train Count':<12} {'Val Count':<10} {'Train %':<8} {'Val %':<8}\n")
        f.write("-"*70 + "\n")
        for pred in sorted(all_predictions):
            train_count = train_results['prediction_distribution'].get(pred, 0)
            val_count = val_results['prediction_distribution'].get(pred, 0)
            train_pct = train_count / train_results['total_samples'] * 100
            val_pct = val_count / val_results['total_samples'] * 100
            f.write(f"{pred:<30} {train_count:<12} {val_count:<10} {train_pct:<7.1f}% {val_pct:<7.1f}%\n")

        
        f.write("\nTRAIN SET FAILED CASES (sample):\n")
        for i, case in enumerate(train_results['failed_cases'][:5]):
            f.write(f"  {i+1}. Expected: '{case['expected']}' | Generated: '{case['generated']}'\n")

        f.write("\nVALIDATION SET FAILED CASES (sample):\n")
        for i, case in enumerate(val_results['failed_cases'][:5]):
            f.write(f"  {i+1}. Expected: '{case['expected']}' | Generated: '{case['generated']}'\n")

    print("Evaluation results saved to evaluation_results.txt")

    
    return {
        'train_accuracy': train_results['accuracy'],
        'val_accuracy': val_results['accuracy'],
        'train_results': train_results,
        'val_results': val_results
    }

def main():
    """主函数：执行完整流程"""
    print("Starting LoRA merge and vLLM evaluation process...")
    
    merged_model_path = './outputs/router_selector_merged'
    
   
    def is_merged_model_valid(path):
        if not os.path.exists(path):
            return False
        
        required_files = ['config.json', 'tokenizer_config.json']
        for file in required_files:
            if not os.path.exists(os.path.join(path, file)):
                return False
       
        weight_files = [f for f in os.listdir(path) if f.endswith(('.bin', '.safetensors'))]
        return len(weight_files) > 0
    
    if is_merged_model_valid(merged_model_path):
        print(f"Found existing valid merged model at {merged_model_path}")
        use_existing = input("Use existing merged model? (y/n): ").lower().strip()
        
        if use_existing != 'y':
            merged_model_path = merge_lora_model()
    else:
        print("No valid merged model found. Creating new merged model...")
        merged_model_path = merge_lora_model()
    
   
    if not is_merged_model_valid(merged_model_path):
        raise RuntimeError(f"Failed to create valid merged model at {merged_model_path}")
    
    print(f"Using merged model from: {merged_model_path}")
    
  
    results = evaluate_with_vllm(merged_model_path)
    
    print(f"\nEvaluation completed!")
    print(f"Train Accuracy: {results['train_accuracy']:.2%}")
    print(f"Validation Accuracy: {results['val_accuracy']:.2%}")
    
    return results

if __name__ == "__main__":
    results = main()