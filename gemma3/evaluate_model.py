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

# =============================================================================
# 配置参数 - 可灵活修改
# =============================================================================
CONFIG = {
    'model_name': 'google/gemma-3-1b-it',
    'train_data_path': 'data/train_samples.pkl',
    'val_data_path': 'data/val_samples.pkl',
    'output_dir': './outputs/gemma-3-1b-it/',
    'max_len': 2048,
    'gpu_id': '0',
    'batch_size': 32,
    'max_tokens': 20,
    'temperature': 0.0,
    'gpu_memory_utilization': 0.8,
    'target_key': 'best_model',  # JSON中要提取的key
    'oracle_field': 'oracle_model_to_route_to',  # 数据中的标准答案字段
}

def build_prompt(sample, candidate_models):
    """构建推理prompt"""
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

    return f"""You are a model selector. Choose ONLY the model name from this list: {candidates_str}

    For the task: {eval_name}
    Tags: {tags}
    Question: {user_prompt}

    Respond in JSON format:
        {{"best_model": ""}}"""


def extract_prediction(generated_text):
    """从模型输出中提取 best_model 字段 - 专注于JSON格式匹配"""
    if not generated_text:
        return "GENERATION_ERROR"

    import json
    import re
    
    # 清理输出 - 移除markdown代码块标记
    cleaned_output = re.sub(r'```json\s*', '', generated_text, flags=re.IGNORECASE)
    cleaned_output = re.sub(r'```\s*', '', cleaned_output)
    cleaned_output = cleaned_output.strip()

    try:
        # 方法1: 直接匹配标准JSON格式 {"best_model": "model_name"}
        json_pattern = r'\{\s*"best_model"\s*:\s*"([^"]+)"\s*\}'
        match = re.search(json_pattern, cleaned_output, re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # 方法2: 尝试解析完整的JSON对象
        json_start = cleaned_output.find('{')
        json_end = cleaned_output.rfind('}') + 1
        if json_start != -1 and json_end > json_start:
            json_str = cleaned_output[json_start:json_end]
            try:
                parsed = json.loads(json_str)
                if 'best_model' in parsed:
                    return str(parsed['best_model']).strip()
            except json.JSONDecodeError:
                pass

        # 方法3: 更宽松的best_model字段匹配
        patterns = [
            r'"best_model"\s*:\s*"([^"]*)"',
            r"'best_model'\s*:\s*'([^']*)'",
            r'best_model["\s]*:["\s]*([^",}\s]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, cleaned_output, re.IGNORECASE)
            if match:
                result = match.group(1).strip().strip('"\'')
                if result:
                    return result

    except Exception:
        pass

    return "GENERATION_ERROR"


def get_candidate_models(data_paths):
    """从数据中提取候选模型"""
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
    """清理GPU内存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    
    # 清理分布式进程组（如果存在）
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
        print(f"✅ Already merged: {merged_path}")
        return merged_path, latest_checkpoint

    print("🔄 Starting model merge process...")
    cleanup_gpu_memory()
    print("💾 Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 设置 pad_token，防止缺失导致 vocab size 不一致
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    
    print("📥 Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager"
    )
    # 调整 base_model 的 vocab size
    base_model.resize_token_embeddings(len(tokenizer))
    print("🔧 Loading LoRA adapter...")
    peft_model = PeftModel.from_pretrained(base_model, lora_path)

    print("🔀 Merging weights...")
    merged_model = peft_model.merge_and_unload()

    print("💾 Saving merged model...")
    os.makedirs(merged_path, exist_ok=True)
    merged_model.save_pretrained(merged_path, safe_serialization=False, max_shard_size="5GB")

    

    # 保存 tokenizer
    tokenizer.save_pretrained(merged_path)

    # 清理资源
    del base_model, peft_model, merged_model, tokenizer
    cleanup_gpu_memory()

    print(f"✅ Model merged to: {merged_path}")
    return merged_path, latest_checkpoint


def initialize_vllm(model_path):
    """初始化vLLM"""
    # 设置环境变量避免多进程问题
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    
    print("🚀 Initializing vLLM...")
    cleanup_gpu_memory()
    
    try:
        llm = LLM(
            model=model_path,
            tensor_parallel_size=1,
            gpu_memory_utilization=CONFIG['gpu_memory_utilization'],
            max_model_len=CONFIG['max_len'],
            dtype="float16",
            trust_remote_code=True,
            enforce_eager=True,  # 禁用CUDA图形优化
        )
        print("✅ vLLM initialized successfully")
        return llm
        
    except Exception as e:
        print(f"❌ vLLM initialization failed: {e}")
        print("🔄 Retrying with reduced memory settings...")
        
        try:
            llm = LLM(
                model=model_path,
                tensor_parallel_size=1,
                gpu_memory_utilization=0.6,  # 降低内存使用
                max_model_len=min(CONFIG['max_len'], 1024),  # 降低最大长度
                dtype="float16",
                trust_remote_code=True,
                enforce_eager=True,
            )
            print("✅ vLLM initialized with reduced settings")
            return llm
            
        except Exception as e2:
            print(f"❌ vLLM initialization still failed: {e2}")
            raise Exception("Failed to initialize vLLM after retries")


def evaluate_dataset(llm, data, dataset_name, candidate_models):
    """使用vLLM评估数据集"""
    print(f"\n📊 Evaluating {dataset_name}: {len(data)} samples")
    
    # 生成prompts
    prompts = [build_prompt(sample, candidate_models) for sample in data]
    expected = [sample[CONFIG['oracle_field']] for sample in data]
    
    # 设置采样参数
    sampling_params = SamplingParams(
        max_tokens=CONFIG['max_tokens'], 
        temperature=CONFIG['temperature'],
        stop=["\n", "</s>", "}"]
    )
    
    # 分批生成
    all_outputs = []
    print("🔄 Generating responses...")
    
    for i in range(0, len(prompts), CONFIG['batch_size']):
        batch_prompts = prompts[i:i+CONFIG['batch_size']]
        batch_num = i // CONFIG['batch_size'] + 1
        total_batches = (len(prompts) + CONFIG['batch_size'] - 1) // CONFIG['batch_size']
        
        try:
            print(f"   Batch {batch_num}/{total_batches}...")
            batch_outputs = llm.generate(batch_prompts, sampling_params)
            all_outputs.extend(batch_outputs)
            
        except Exception as e:
            print(f"❌ Batch {batch_num} failed: {e}")
            # 创建虚拟输出
            for _ in batch_prompts:
                class DummyOutput:
                    def __init__(self):
                        self.outputs = [type('obj', (object,), {'text': 'ERROR'})]
                all_outputs.append(DummyOutput())
    
    # 处理结果
    print("🔍 Processing results...")
    correct = 0
    
    for idx, (output, true_answer, sample) in enumerate(tqdm(zip(all_outputs, expected, data), total=len(data), desc="Processing")):
        try:
            generated_text = output.outputs[0].text.strip()
            predicted = extract_prediction(generated_text) if generated_text != 'ERROR' else 'GENERATION_ERROR'
        except:
            predicted = 'GENERATION_ERROR'
            generated_text = 'ERROR'
        
        # 精确匹配预测结果和真实答案
        is_correct = (predicted == true_answer)
        if is_correct:
            correct += 1
        
        # 将预测结果写入数据
        sample.update({
            "prediction": predicted,
            "is_correct": is_correct,
            "raw_generated_text": generated_text,
            "evaluation_timestamp": datetime.now().isoformat()
        })
    
    accuracy = correct / len(data) if len(data) > 0 else 0
    print(f"✅ {dataset_name} Accuracy: {accuracy:.2%} ({correct}/{len(data)})")
    
    # 显示前几个样本的结果
    print(f"\n📋 Sample Results from {dataset_name}:")
    for i, sample in enumerate(data[:3]):
        print(f"Sample {i+1}:")
        print(f"  Expected: {sample.get(CONFIG['oracle_field'])}")
        print(f"  Predicted: {sample.get('prediction')}")
        print(f"  Correct: {sample.get('is_correct')}")
        print(f"  Raw: {sample.get('raw_generated_text')[:80]}...")
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
    
    # 更新配置
    CONFIG.update({k: v for k, v in vars(args).items() if k in CONFIG})
    
    # 设置GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = CONFIG['gpu_id']
    
    print("🚀 Starting vLLM Evaluation...")
    print(f"📱 Model: {CONFIG['model_name']}")
    print(f"📁 Output: {CONFIG['output_dir']}")
    print(f"🎯 Max tokens: {CONFIG['max_tokens']}")
    print(f"🌡️ Temperature: {CONFIG['temperature']}")
    print(f"📦 Batch size: {CONFIG['batch_size']}")
    
    # 获取候选模型
    data_paths = [CONFIG['train_data_path']]
    if args.use_validation:
        data_paths.append(CONFIG['val_data_path'])
    candidate_models = get_candidate_models(data_paths)
    print(f"🎯 Found {len(candidate_models)} candidate models")
    
    # 检查或创建合并模型
    merged_path = f'{CONFIG["output_dir"].rstrip("/")}_merged'
    if not os.path.exists(merged_path) or args.force_merge:
        merged_path, checkpoint = merge_lora_model(CONFIG['output_dir'], CONFIG['model_name'])
    else:
        print(f"✅ Using existing merged model: {merged_path}")
        checkpoint = "existing"
    
    # 初始化vLLM
    llm = initialize_vllm(merged_path)
    
    try:
        # 评估训练集
        train_data = load_dataset(CONFIG['train_data_path'])
        train_results = evaluate_dataset(llm, train_data, "Train", candidate_models)
        
        # 评估验证集（如果指定）
        val_results = None
        if os.path.exists(CONFIG['val_data_path']):
            val_data = load_dataset(CONFIG['val_data_path'])
            val_results = evaluate_dataset(llm, val_data, "Validation", candidate_models)
    
    finally:
        # 清理资源
        print("🧹 Cleaning up resources...")
        if llm:
            del llm
        cleanup_gpu_memory()
    
    # 保存结果
    timestamp = datetime.now().strftime("%m%d_%H%M")
    
    print("💾 Saving results...")
    # 备份并更新训练数据
    backup_train = CONFIG['train_data_path'].replace('.pkl', f'_backup_{timestamp}.pkl')
    shutil.copy(CONFIG['train_data_path'], backup_train)
    
    with open(CONFIG['train_data_path'], "wb") as f:
        pickle.dump(train_results['modified_data'], f)
    print(f"✅ Updated {CONFIG['train_data_path']}")
    
    # 备份并更新验证数据（如果有）
    if val_results:
        backup_val = CONFIG['val_data_path'].replace('.pkl', f'_backup_{timestamp}.pkl')
        shutil.copy(CONFIG['val_data_path'], backup_val)
        with open(CONFIG['val_data_path'], "wb") as f:
            pickle.dump(val_results['modified_data'], f)
        print(f"✅ Updated {CONFIG['val_data_path']}")
    
    # 保存评估报告
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
    
    # 打印最终结果
    print("\n" + "="*60)
    print("🎉 vLLM EVALUATION COMPLETED!")
    print("="*60)
    print(f"📊 Train Accuracy: {train_results['accuracy']:.2%}")
    if val_results:
        print(f"📊 Val Accuracy: {val_results['accuracy']:.2%}")
        overfitting = train_results['accuracy'] - val_results['accuracy']
        if overfitting > 0.05:
            print(f"⚠️ Overfitting: {overfitting:+.2%}")
        else:
            print(f"✅ Overfitting: {overfitting:+.2%}")
    print(f"🎯 Candidate Models: {len(candidate_models)}")
    print(f"📄 Report saved: {report_path}")
    print("="*60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 程序被用户中断")
    except Exception as e:
        print(f"\n❌ 程序执行出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 最终清理
        cleanup_gpu_memory()
        print("🧹 程序结束，已清理资源")