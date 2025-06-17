import os
import shutil
import torch
import pickle
import argparse
import json
import re
import gc
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from dataset_loader import load_dataset
from tqdm import tqdm
from datetime import datetime

CONFIG = {
    'model_name': 'google/gemma-3-4b-it',
    'train_data_path': 'data/train_samples_4b_multi.pkl',
    'val_data_path': 'data/val_samples_4b_multi.pkl',
    'output_dir': './outputs/gemma-3-4b-it_multiclass/',
    'max_len': 2048,
    'gpu_id': '2',
    'batch_size': 32,
    'max_tokens': 100,
    'temperature': 0.7,
    'gpu_memory_utilization': 0.8,
    'target_key': 'best_model',  # JSON
    'oracle_field': 'oracle_model_to_route_to', 
}

def build_prompt(sample, candidate_models):
    """构建推理prompt - 支持多模型选择"""
    candidates_str = ", ".join(sorted(candidate_models))
    eval_name = sample.get("eval_name", "N/A")
    user_prompt = sample.get("prompt", "")
    
    # 处理user_prompt
    if isinstance(user_prompt, list):
        user_prompt = ' '.join(str(p) for p in user_prompt)
    elif not isinstance(user_prompt, str):
        user_prompt = str(user_prompt)
    
    # 截断过长的prompt
    user_prompt = user_prompt[:1800]
    
    tags = ", ".join(sample.get("tags", [])) if isinstance(sample.get("tags", []), list) else str(sample.get("tags", ""))

    return f"""You are a model selector. Choose the best model(s) from this list: {candidates_str}

    For the task: {eval_name}
    Tags: {tags}
    Question: {user_prompt}

    You can select multiple models if they are equally good. For each selected model, output a separate JSON object:
    {{"best_model": "model_name"}}
    If multiple models are good, output like: {{"best_model": "model1"}} {{"best_model": "model2"}}

    IMPORTANT: If NO model from the list is suitable for this task, output:
    {{"best_model": "no_model_correct"}}"""

def is_prediction_correct(predictions, expected_models):
        """根据ground truth数量自适应比较预测结果"""
        # 特殊情况：ground truth为空时，期望预测"no_model_correct"
        if not expected_models:
            return predictions == ["no_model_correct"]
        
        # 特殊情况：如果预测了"no_model_correct"但有期望模型
        if "no_model_correct" in predictions:
            return False
        
       
        if not predictions:
            return False
        
       
        expected_count = len(expected_models)
        predictions_to_compare = predictions[:expected_count]
        
     
        if len(predictions_to_compare) != expected_count:
            return False
        
     
        predictions_clean = set(pred.lower().strip() for pred in predictions_to_compare)
        expected_clean = set(exp.lower().strip() for exp in expected_models)
        
        return predictions_clean == expected_clean

def extract_ground_truth(sample, start_col=3, end_col=14):
   
 
    if hasattr(sample, 'keys'):
        all_columns = list(sample.keys())
    elif hasattr(sample, 'index'):  
        all_columns = list(sample.index)
    else:
        return []
    

    model_columns = all_columns[start_col:end_col+1] if len(all_columns) > end_col else all_columns[start_col:]
    

    model_scores = {}
    ground_truth_models = []
    
    for model_name in model_columns:
        try:
            score = sample[model_name] if hasattr(sample, '__getitem__') else getattr(sample, model_name, 0)
            score = float(score) if score is not None else 0.0
            model_scores[model_name] = score
            
            # 如果分数为1，加入ground truth
            if score == 1.0:
                ground_truth_models.append(model_name)
        except:
            model_scores[model_name] = 0.0
    
    # 如果没有分数为1的模型，从小于1的分数中选择最高的
    if not ground_truth_models:
        # 过滤出分数大于0且小于1的模型
        valid_models = {name: score for name, score in model_scores.items() if 0 < score < 1}
        
        if valid_models:
            # 按分数排序，取最高的，最多3个
            sorted_models = sorted(valid_models.items(), key=lambda x: x[1], reverse=True)
            ground_truth_models = [model_name for model_name, score in sorted_models[:3]]
        # 如果没有分数大于0的模型，返回空列表
    
    
    return ground_truth_models

def extract_prediction(raw_output):
       
        if not raw_output:
            return []
        
        import re
        
        # 查找所有 "best_model": "value" 模式，不管JSON是否完整
        pattern = r'"best_model"\s*:\s*"([^"]*)'  
        matches = re.findall(pattern, raw_output, re.IGNORECASE)
        
        models = []
        for match in matches:
            model = match.strip()
            if model:
                models.append(model)
        
        print(f"Extracted models: {models}")
        return models


def get_candidate_models(data_paths):
  
    all_data = []
    for path in data_paths:
        if os.path.exists(path):
            all_data.extend(load_dataset(path))
    return sorted(list(set(ex[CONFIG['oracle_field']] for ex in all_data)))

def find_latest_checkpoint(output_dir):
    """查找最新的checkpoint"""
    checkpoint_dirs = [d for d in os.listdir(output_dir) if d.startswith('checkpoint-')]
    if not checkpoint_dirs:
        raise FileNotFoundError(f"No checkpoints found in {output_dir}")
    checkpoint_dirs.sort(key=lambda x: int(x.split('-')[1]))
    return checkpoint_dirs[-1]

def cleanup_gpu_memory():

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    
    try:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    except:
        pass

def merge_lora_model(output_dir, model_name):
    """合并LoRA模型"""
    latest_checkpoint = find_latest_checkpoint(output_dir)
    lora_path = os.path.join(output_dir, latest_checkpoint)
    merged_path = f'{output_dir.rstrip("/")}_merged'

    print(f"Using checkpoint: {latest_checkpoint}")

    if os.path.exists(merged_path):
        print(f"Already merged: {merged_path}")
        return merged_path, latest_checkpoint

    print("Starting model merge process...")
    cleanup_gpu_memory()
    print("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

   
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager"
    )
 
    base_model.resize_token_embeddings(len(tokenizer))
    print("Loading LoRA adapter...")
    peft_model = PeftModel.from_pretrained(base_model, lora_path)

    print("Merging weights...")
    merged_model = peft_model.merge_and_unload()

    print("Saving merged model...")
    os.makedirs(merged_path, exist_ok=True)
    merged_model.save_pretrained(merged_path, safe_serialization=False, max_shard_size="5GB")

    tokenizer.save_pretrained(merged_path)

    
    del base_model, peft_model, merged_model, tokenizer
    cleanup_gpu_memory()

    print(f"Model merged to: {merged_path}")
    return merged_path, latest_checkpoint


def initialize_vllm(model_path):
    
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    
    print("Initializing vLLM...")
    cleanup_gpu_memory()
    
    try:
        llm = LLM(
            model=model_path,
            tensor_parallel_size=1,
            gpu_memory_utilization=CONFIG['gpu_memory_utilization'],
            max_model_len=CONFIG['max_len'],
            dtype="float32",
            trust_remote_code=True,
            enforce_eager=True,  
        )
        print("vLLM initialized successfully")
        return llm
        
    except Exception as e:
        print(f"vLLM initialization failed: {e}")
        print("Retrying with reduced memory settings...")
        
        try:
            llm = LLM(
                model=model_path,
                tensor_parallel_size=1,
                gpu_memory_utilization=0.6,  
                max_model_len=CONFIG['max_len'],  
                dtype="float32",
                trust_remote_code=True,
                enforce_eager=True,
            )
            print("vLLM initialized with reduced settings")
            return llm
            
        except Exception as e2:
            print(f"vLLM initialization still failed: {e2}")
            raise Exception("Failed to initialize vLLM after retries")


def evaluate_dataset(llm, data, dataset_name, candidate_models):
    """使用vLLM评估数据集 - 支持多模型预测"""
    print(f"\nEvaluating {dataset_name}: {len(data)} samples")
    

    prompts = [build_prompt(sample, candidate_models) for sample in data]
    
    
    sampling_params = SamplingParams(
        max_tokens=CONFIG['max_tokens'], 
        temperature=CONFIG['temperature'],
        stop=["\n\n", "</s>"]  
    )
    
    # 分批生成
    all_outputs = []
    print("Generating responses...")
    
    for i in range(0, len(prompts), CONFIG['batch_size']):
        batch_prompts = prompts[i:i+CONFIG['batch_size']]
        batch_num = i // CONFIG['batch_size'] + 1
        total_batches = (len(prompts) + CONFIG['batch_size'] - 1) // CONFIG['batch_size']
        
        try:
            print(f"   Batch {batch_num}/{total_batches}...")
            batch_outputs = llm.generate(batch_prompts, sampling_params)
            all_outputs.extend(batch_outputs)
            
        except Exception as e:
            print(f"Batch {batch_num} failed: {e}")
          
            for _ in batch_prompts:
                class DummyOutput:
                    def __init__(self):
                        self.outputs = [type('obj', (object,), {'text': 'ERROR'})]
                all_outputs.append(DummyOutput())
    
  
    print("Processing results...")
    correct = 0
    
    for idx, (output, sample) in enumerate(tqdm(zip(all_outputs, data), total=len(data), desc="Processing")):
        try:
            generated_text = output.outputs[0].text.strip()
            predicted_models = extract_prediction(generated_text) if generated_text != 'ERROR' else ["GENERATION_ERROR"]
        except:
            predicted_models = ["GENERATION_ERROR"]
            generated_text = 'ERROR'
        
        
        expected_models = extract_ground_truth(sample)
        
      
        is_correct = is_prediction_correct(predicted_models, expected_models)
        if is_correct:
            correct += 1
   
        sample.update({
            "prediction": predicted_models,  
            "expected_models": expected_models,  
            "is_correct": is_correct,
            "raw_generated_text": generated_text,
            "evaluation_timestamp": datetime.now().isoformat()
        })
    
    accuracy = correct / len(data) if len(data) > 0 else 0
    print(f"{dataset_name} Accuracy: {accuracy:.2%} ({correct}/{len(data)})")
    
    
    print(f"\nSample Results from {dataset_name}:")
    for i, sample in enumerate(data[:3]):
        print(f"Sample {i+1}:")
        print(f"  Expected: {sample.get('expected_models')}")
        print(f"  Predicted: {sample.get('prediction')}")
        print(f"  Correct: {sample.get('is_correct')}")
        print(f"  Raw: {sample.get('raw_generated_text')}...")
        print()

    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': len(data),
        'modified_data': data
    }


def main():
    parser = argparse.ArgumentParser(description="vLLM Model Evaluation")
    for key, value in CONFIG.items():
        parser.add_argument(f'--{key}', default=value, type=type(value))
    parser.add_argument('--use_validation', action='store_true', help='Also evaluate validation set')
    parser.add_argument('--force_merge', action='store_true', help='Force re-merge LoRA model')
    args = parser.parse_args()
    
   
    CONFIG.update({k: v for k, v in vars(args).items() if k in CONFIG})
 
    os.environ["CUDA_VISIBLE_DEVICES"] = CONFIG['gpu_id']
    
    print("Starting vLLM Evaluation...")
    print(f"Model: {CONFIG['model_name']}")
    print(f"Output: {CONFIG['output_dir']}")
    print(f"Max tokens: {CONFIG['max_tokens']}")
    print(f"Temperature: {CONFIG['temperature']}")
    print(f"Batch size: {CONFIG['batch_size']}")
    
    
    data_paths = [CONFIG['train_data_path']]
    if args.use_validation:
        data_paths.append(CONFIG['val_data_path'])
    candidate_models = get_candidate_models(data_paths)
    print(f"Found {len(candidate_models)} candidate models")
    
  
    merged_path = f'{CONFIG["output_dir"].rstrip("/")}_merged'
    if not os.path.exists(merged_path) or args.force_merge:
        merged_path, checkpoint = merge_lora_model(CONFIG['output_dir'], CONFIG['model_name'])
    else:
        print(f"Using existing merged model: {merged_path}")
        checkpoint = "existing"
    
    
    llm = initialize_vllm(merged_path)
    
    try:
       
        train_data = load_dataset(CONFIG['train_data_path'])
        train_results = evaluate_dataset(llm, train_data, "Train", candidate_models)
        
        # 评估验证集（如果指定）
        val_results = None
        if os.path.exists(CONFIG['val_data_path']):
            val_data = load_dataset(CONFIG['val_data_path'])
            val_results = evaluate_dataset(llm, val_data, "Validation", candidate_models)
    
    finally:   
        print("Cleaning up resources...")
        if llm:
            del llm
        cleanup_gpu_memory()
    
 
    timestamp = datetime.now().strftime("%m%d_%H%M")
    
    print("Saving results...")
  
    backup_train = CONFIG['train_data_path'].replace('.pkl', f'_backup_{timestamp}.pkl')
    shutil.copy(CONFIG['train_data_path'], backup_train)
    
    with open(CONFIG['train_data_path'], "wb") as f:
        pickle.dump(train_results['modified_data'], f)
    print(f"Updated {CONFIG['train_data_path']}")
    
    if val_results:
        backup_val = CONFIG['val_data_path'].replace('.pkl', f'_backup_{timestamp}.pkl')
        shutil.copy(CONFIG['val_data_path'], backup_val)
        with open(CONFIG['val_data_path'], "wb") as f:
            pickle.dump(val_results['modified_data'], f)
        print(f"Updated {CONFIG['val_data_path']}")
    
    report_path = os.path.join(CONFIG['output_dir'], f"evaluation_report_{timestamp}.txt")
    with open(report_path, "w") as f:
        f.write(f"vLLM EVALUATION REPORT\n")
        f.write(f"=====================\n")
        f.write(f"Model: {CONFIG['model_name']}\n")
        f.write(f"Checkpoint: {checkpoint}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Candidate Models: {len(candidate_models)}\n")
        f.write(f"Max Tokens: {CONFIG['max_tokens']}\n")
        f.write(f"Temperature: {CONFIG['temperature']}\n")
        f.write(f"Batch Size: {CONFIG['batch_size']}\n\n")
        f.write(f"RESULTS:\n")
        f.write(f"Train Accuracy: {train_results['accuracy']:.2%} ({train_results['correct']}/{train_results['total']})\n")
        if val_results:
            f.write(f"Val Accuracy: {val_results['accuracy']:.2%} ({val_results['correct']}/{val_results['total']})\n")
            f.write(f"Overfitting: {train_results['accuracy'] - val_results['accuracy']:+.2%}\n")
        f.write(f"\nCandidate Models:\n")
        for i, model in enumerate(candidate_models, 1):
            f.write(f"{i:2d}. {model}\n")
    

    print("\n" + "="*60)
    print("vLLM EVALUATION COMPLETED!")
    print("="*60)
    print(f"Train Accuracy: {train_results['accuracy']:.2%}")
    if val_results:
        print(f"Val Accuracy: {val_results['accuracy']:.2%}")
        overfitting = train_results['accuracy'] - val_results['accuracy']
        if overfitting > 0.05:
            print(f"⚠️ Overfitting: {overfitting:+.2%}")
        else:
            print(f"Overfitting: {overfitting:+.2%}")
    print(f"Candidate Models: {len(candidate_models)}")
    print(f"Report saved: {report_path}")
    print("="*60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"\n程序执行出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 最终清理
        cleanup_gpu_memory()
        print("程序结束，已清理资源")