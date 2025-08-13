import pickle
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict
import re
from tqdm import tqdm
import os
from datetime import datetime
import argparse
import time
import random
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4"
os.environ["PYTORCH_USE_CUDA_DSA"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
class PIITaggingDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=2048):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        # self.template = (
        #     "You are a helpful assistant. Please identify tags of user intentions "
        #     "in the following user query and provide an explanation for each tag. "
        #     "Please respond in the JSON format {{\"tag\": str, \"explanation\": str}}. "
        #     "Query: {query} Assistant: {response}"
        # )
        self.template = (
            "You are a helpful assistant. For the user query below, generate tags in this order: "
            "1) Domain, 2) Task Type, 3) Difficulty, 4) Language, 5) Topics (can be multiple). "
            "Explain each tag briefly. Output must be JSON: {{\"tag\": str, \"explanation\": str}}. "
            "Query: {query} Assistant: {response}"

        )

        
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        query = item['prompt']
        
        # 处理tags
        if isinstance(item['parsed_tags'], str):
            try:
                tags = eval(item['parsed_tags'])
            except:
                tags = item['parsed_tags']
        else:
            tags = item['parsed_tags']
        
        if not isinstance(tags, list):
            tags = [tags]
        
        response = json.dumps(tags, ensure_ascii=False)
        
        # 确保response以EOS token结尾
        if self.tokenizer.eos_token:
            response = response + self.tokenizer.eos_token
        
        # 裁剪query到最大1800 tokens
        query_tokens = self.tokenizer.encode(query, add_special_tokens=False)
        if len(query_tokens) > 1800:
            query_tokens = query_tokens[:1800]
            query = self.tokenizer.decode(query_tokens, skip_special_tokens=True)
        
        text = self.template.format(query=query, response=response)
        
        # 分别编码prompt和完整文本
        prompt_text = self.template.format(query=query, response="").rstrip()
        
        prompt_encoding = self.tokenizer(prompt_text, add_special_tokens=False)
        full_encoding = self.tokenizer(
            text, truncation=True, max_length=self.max_length,
            padding='max_length', return_tensors="pt"
        )
        
        labels = full_encoding['input_ids'].clone()
        
        # 将prompt部分设为-100
        prompt_length = len(prompt_encoding['input_ids'])
        labels[0, :prompt_length] = -100
        # padding部分也设为-100
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        # 调试：检查有效labels
        valid_labels = (labels != -100).sum().item()
        total_labels = labels.numel()
        if valid_labels == 0:
            print(f"ERROR: Sample {idx} has no valid labels!")
            print(f"Prompt length: {prompt_length}, Total length: {total_labels}")
            print(f"Query: {query[:100]}...")
            # 强制保留一些labels
            labels[0, -50:] = full_encoding['input_ids'][0, -50:].clone()
        
        return {
            'input_ids': full_encoding['input_ids'].squeeze(),
            'attention_mask': full_encoding['attention_mask'].squeeze(),
            'labels': labels.squeeze(),
        }

class TaggingEvaluator:
    def __init__(self, device='cuda'):
        self.phrase_model = SentenceTransformer('whaleloops/phrase-bert')
        self.phrase_model = self.phrase_model.to(device)
        
    def extract_tags_from_response(self, response: str) -> List[Dict[str, str]]:
        response = response.strip()
        result = []
        
        try:
            # 方法1: 尝试解析为JSON数组
            if response.startswith('['):
                try:
                    result = json.loads(response)
                    return result
                except json.JSONDecodeError:
                    # 尝试修复截断的JSON数组
                    bracket_count = 0
                    end_pos = -1
                    for i, char in enumerate(response):
                        if char == '[':
                            bracket_count += 1
                        elif char == ']':
                            bracket_count -= 1
                            if bracket_count == 0:
                                end_pos = i + 1
                                break
                    if end_pos > 0:
                        first_json = response[:end_pos]
                        result = json.loads(first_json)
                        return result
            
            # 方法2: 处理多个独立的JSON对象（当前情况）
            # 查找所有完整的JSON对象
            pattern = r'\{"tag":\s*"([^"]+)",\s*"explanation":\s*"([^"]*?)"\}'
            matches = re.findall(pattern, response, re.DOTALL)
            
            if matches:
                result = [{"tag": tag, "explanation": exp} for tag, exp in matches]
                return result
            
            # 方法3: 逐个提取完整的JSON对象
            json_objects = []
            current_pos = 0
            
            while current_pos < len(response):
                # 查找下一个JSON对象的开始
                start_pos = response.find('{"tag":', current_pos)
                if start_pos == -1:
                    break
                
                # 查找这个JSON对象的结束
                brace_count = 0
                end_pos = -1
                for i in range(start_pos, len(response)):
                    if response[i] == '{':
                        brace_count += 1
                    elif response[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_pos = i + 1
                            break
                
                if end_pos > start_pos:
                    try:
                        json_str = response[start_pos:end_pos]
                        json_obj = json.loads(json_str)
                        json_objects.append(json_obj)
                    except json.JSONDecodeError:
                        pass
                    
                    current_pos = end_pos
                else:
                    break
            
            if json_objects:
                return json_objects
                
        except Exception:
            pass
            
        return []
    
    def calculate_exact_match_f1(self, pred_tags: List[str], gold_tags: List[str]) -> float:
        pred_set = set(pred_tags)
        gold_set = set(gold_tags)
        
        if len(pred_set) == 0 and len(gold_set) == 0:
            return 1.0
        if len(pred_set) == 0 or len(gold_set) == 0:
            return 0.0
        
        intersection = pred_set & gold_set
        precision = len(intersection) / len(pred_set)
        recall = len(intersection) / len(gold_set)
        
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)
    
    def calculate_semantic_f1(self, pred_tags: List[str], gold_tags: List[str], threshold=0.8) -> float:
        if len(pred_tags) == 0 and len(gold_tags) == 0:
            return 1.0
        if len(pred_tags) == 0 or len(gold_tags) == 0:
            return 0.0
        
        pred_embeddings = self.phrase_model.encode(pred_tags)
        gold_embeddings = self.phrase_model.encode(gold_tags)
        
        # 使用余弦相似度而非点积
        from sklearn.metrics.pairwise import cosine_similarity
        similarity_matrix = cosine_similarity(pred_embeddings, gold_embeddings)
        
        # 每个预测标签找最佳匹配
        matched_pred = 0
        for i in range(len(pred_tags)):
            if np.max(similarity_matrix[i]) >= threshold:
                matched_pred += 1
        
        # 每个真实标签找最佳匹配        
        matched_gold = 0
        for j in range(len(gold_tags)):
            if np.max(similarity_matrix[:, j]) >= threshold:
                matched_gold += 1
        
        precision = matched_pred / len(pred_tags)
        recall = matched_gold / len(gold_tags)
        
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

def load_data(pkl_path: str, limit_data: int = None):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    processed_data = []
    for item in data:
        if isinstance(item, dict) and 'prompt' in item and 'parsed_tags' in item:
            processed_data.append(item)
        elif isinstance(item, (list, tuple)) and len(item) >= 3:
            processed_data.append({
                'index': item[0],
                'prompt': item[1],
                'parsed_tags': item[2]
            })
    
    # 限制数据量
    if limit_data is not None and limit_data > 0:
        processed_data = processed_data[:limit_data]
    
    return processed_data

def evaluate_model(model, tokenizer, eval_dataset, evaluator, device, show_samples=True):
    model.eval()
    exact_match_f1_scores = []
    semantic_f1_scores = []
    if isinstance(model, nn.DataParallel):
        generate_model = model.module
    else:
        generate_model = model
    eval_count = min(3, len(eval_dataset))
    with torch.no_grad():
        for i in tqdm(range(eval_count), desc="Evaluating"):
            query = eval_dataset.data[i]['prompt']
            query_prompt = eval_dataset.template.format(query=query, response="").rstrip()
            
            inputs = tokenizer(query_prompt, return_tensors="pt").to(device)
            outputs = generate_model.generate(
                **inputs, 
                max_new_tokens=512, 
                do_sample=True, 
                temperature=0.6, 
                top_p=0.9, 
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.2
            )
            
            response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            
            # 提取预测tags
            pred_tags_dicts = evaluator.extract_tags_from_response(response)
            pred_tags = [tag_dict.get('tag', '') for tag_dict in pred_tags_dicts if isinstance(tag_dict, dict) and 'tag' in tag_dict]
            
            # 直接从原始数据获取gold tags
            gold_tags_dicts = eval_dataset.data[i]['parsed_tags']
            if isinstance(gold_tags_dicts, str):
                try:
                    gold_tags_dicts = eval(gold_tags_dicts)
                except:
                    gold_tags_dicts = []
            
            if not isinstance(gold_tags_dicts, list):
                gold_tags_dicts = [gold_tags_dicts]
            
            # 从gold_tags_dicts中提取tag名称
            gold_tags = [tag_dict.get('tag', '') for tag_dict in gold_tags_dicts 
                        if isinstance(tag_dict, dict) and 'tag' in tag_dict]
            
            # 显示前3个样本的详细信息
            if show_samples and i < 3:
                print(f"\n{'='*80}")
                print(f"SAMPLE {i+1}:")
                print(f"{'='*80}")
                print(f"QUERY: {query}")
                print(f"\nGROUND TRUTH TAGS: {gold_tags}")
                print(f"GROUND TRUTH FULL: {gold_tags_dicts}")
                print(f"\nMODEL RESPONSE:")
                print("```json")
                if pred_tags_dicts:
                    print(json.dumps(pred_tags_dicts, indent=2, ensure_ascii=False))
                    print("```")
                    print(f"\nPREDICTED TAGS: {pred_tags}")
                    print(f"PREDICTED FULL: {pred_tags_dicts}")
                else:
                    print("解析失败，原始输出:")
                    print(response[:500] + "..." if len(response) > 500 else response)
                    print("```")
                    print(f"\nPREDICTED TAGS: {pred_tags}")
                    print(f"PREDICTED FULL: {pred_tags_dicts}")
                
                # 计算这个样本的指标
                em_f1 = evaluator.calculate_exact_match_f1(pred_tags, gold_tags)
                sem_f1 = evaluator.calculate_semantic_f1(pred_tags, gold_tags)
                print(f"\nSAMPLE METRICS:")
                print(f"  - Exact Match F1: {em_f1:.3f}")
                print(f"  - Semantic F1: {sem_f1:.3f}")
                print(f"{'='*80}")
            
            # 计算指标
            em_f1 = evaluator.calculate_exact_match_f1(pred_tags, gold_tags)
            sem_f1 = evaluator.calculate_semantic_f1(pred_tags, gold_tags)
            
            exact_match_f1_scores.append(em_f1)
            semantic_f1_scores.append(sem_f1)
    
    return {
        'exact_match_f1': np.mean(exact_match_f1_scores) if exact_match_f1_scores else 0.0,
        'semantic_f1': np.mean(semantic_f1_scores) if semantic_f1_scores else 0.0
    }
def main():
    parser = argparse.ArgumentParser(description='Fine-tune INSTAGGER models')
    parser.add_argument('--limit_data', type=int, default=None, 
                       help='Limit the number of data samples for testing (e.g., 100)')
    parser.add_argument('--pkl_path', type=str, default='pii_result.pkl',
                       help='Path to the pickle file containing data')
    parser.add_argument('--models', nargs='+', 
                       default=["OFA-Sys/InsTagger", "google/gemma-2-2b"],
                       help='List of model names to fine-tune')
    parser.add_argument('--batch_size', type=int, default=2,
                       help='Per device batch size (default: 2)')
    parser.add_argument('--epochs', type=int, default=1,
                       help='Number of training epochs (default: 1)')
    
    
    # 数据划分选项
    parser.add_argument('--use_all_data', action='store_true',
                       help='Use all data for training (no train/eval split)')
    parser.add_argument('--eval_split', type=float, default=0.2,
                       help='Fraction of data to use for evaluation (default: 0.2)')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()
    print(f"Using device: {device}")
    print(f"Number of GPUs: {num_gpus}")
    
    # 加载数据
    print("Loading data...")
    data = load_data(args.pkl_path, args.limit_data)
    print(f"Loaded {len(data)} samples")
    if args.limit_data:
        print(f"Data limited to {args.limit_data} samples for testing")
    
    # 划分数据集
    if args.use_all_data:
        print("Using ALL data for training (no validation split)")
        train_data = data
        # 随机取样用于评估显示
        eval_sample_size = min(50, len(data))
        eval_data = random.sample(data, eval_sample_size)
        print(f"Train samples: {len(train_data)} (100%), Eval samples: {len(eval_data)} (random sample for display only)")
    else:
        split_idx = int((1 - args.eval_split) * len(data))
        train_data = data[:split_idx]
        eval_data = data[split_idx:]
        print(f"Train samples: {len(train_data)} ({(1-args.eval_split)*100:.0f}%), Eval samples: {len(eval_data)} ({args.eval_split*100:.0f}%)")
    
    evaluator = TaggingEvaluator(device)
    
    for model_name in args.models:
        print(f"\n{'='*50}")
        print(f"Fine-tuning {model_name}")
        
        # 创建目录
        clean_name = model_name.replace("/", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = f"./experiments/{clean_name}_{timestamp}"
        if args.limit_data:
            model_dir += f"_limit{args.limit_data}"
        
        os.makedirs(f"{model_dir}/results", exist_ok=True)
        print(f"Model directory: {model_dir}")
        
        # 加载模型
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto",trust_remote_code=True, attn_implementation="sdpa")
        if hasattr(model, 'generation_config'):
            model.generation_config.do_sample = True
        # 准备数据
        train_dataset = PIITaggingDataset(train_data, tokenizer)
        eval_dataset = PIITaggingDataset(eval_data, tokenizer)
        
        # 训练配置
        per_device_batch_size = args.batch_size
        gradient_accumulation_steps = 4  # 减少accumulation steps
        
        # 根据数据量调整训练步数
        total_steps = len(train_dataset) // (per_device_batch_size * gradient_accumulation_steps)
        if total_steps == 0:
            gradient_accumulation_steps = max(1, len(train_dataset) // per_device_batch_size)
            total_steps = max(1, len(train_dataset) // (per_device_batch_size * gradient_accumulation_steps))

        eval_steps = max(5, total_steps // 5) if total_steps > 5 else total_steps
        save_steps = 100
        logging_steps = max(1, total_steps // 10) if total_steps > 10 else 1

        # 计算训练时间预估 - 仅用于显示基本信息
        print(f"Training config:")
        print(f"  - Per device batch size: {per_device_batch_size}")
        print(f"  - Gradient accumulation steps: {gradient_accumulation_steps}")
        print(f"  - Effective batch size: {per_device_batch_size * gradient_accumulation_steps}")
        print(f"  - Total steps: {total_steps}")
        print(f"  - Total epochs: {args.epochs}")
        print(f"  - Eval steps: {eval_steps}")
        print(f"  - Training will use: {num_gpus} GPU(s)")
        print(f"  - Progress bar will show ETA (estimated time remaining)")
        
        training_args = TrainingArguments(
            output_dir=f"{model_dir}/checkpoints",
            num_train_epochs=args.epochs,
            per_device_train_batch_size=per_device_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=2e-5,
            weight_decay=0.01,
            warmup_steps=max(10, total_steps // 20),
            
            # 日志和保存设置
            save_safetensors=False,
            logging_dir=f"{model_dir}/tensorboard_logs",
            logging_steps=logging_steps,
            save_steps=save_steps,
            eval_strategy="no",           
            save_strategy="steps",        
            load_best_model_at_end=False, 
            
            # 启用tensorboard
            report_to="tensorboard",
            
            # 性能设置
            bf16=True,
            dataloader_pin_memory=True,  
            remove_unused_columns=False,
            gradient_checkpointing=True,  
            save_total_limit=3,
            
            # 进度条和日志设置
            disable_tqdm=False,           
            log_level="info",             
            logging_first_step=True,     
            logging_nan_inf_filter=True,  
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
            processing_class=tokenizer,
        )
        
        # 训练前评估
        print("Evaluating before training...")
        before_metrics = evaluate_model(model, tokenizer, eval_dataset, evaluator, device, show_samples=True)
        print(f"Before - EM F1: {before_metrics['exact_match_f1']:.3f}, Semantic F1: {before_metrics['semantic_f1']:.3f}")

        # 训练
        print(f"\n{'='*60}")
        print(f"Starting training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Transformers trainer will show progress with ETA")
        print(f"Model will be saved every {save_steps} steps")
        print(f"{'='*60}")
        
        start_time = time.time()
        trainer.train()
        end_time = time.time()
        
        actual_training_time = end_time - start_time
        actual_hours = int(actual_training_time // 3600)
        actual_minutes = int((actual_training_time % 3600) // 60)
        actual_seconds = int(actual_training_time % 60)
        
        print(f"\n{'='*60}")
        print(f"Training completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total training time: {actual_hours}h {actual_minutes}m {actual_seconds}s")
        print(f"Average time per step: {actual_training_time/total_steps:.2f}s")
        print(f"{'='*60}")

        # 训练结束后显式保存模型和tokenizer
        print("Saving final model and tokenizer...")
        final_save_path = os.path.join(model_dir, "final_model")
        trainer.save_model(final_save_path)  # 保存模型权重和配置
        tokenizer.save_pretrained(final_save_path)  # 保存tokenizer
        print(f"Final model and tokenizer saved to: {final_save_path}")


        # 训练后评估
        print("Evaluating after training...")
        after_metrics = evaluate_model(model, tokenizer, eval_dataset, evaluator, device, show_samples=True)
        print(f"After - EM F1: {after_metrics['exact_match_f1']:.3f}, Semantic F1: {after_metrics['semantic_f1']:.3f}")
        # 保存结果
        results = {
            "model": model_name,
            "data_info": {
                "total_samples": len(data),
                "train_samples": len(train_data),
                "eval_samples": len(eval_data),
                "limit_data": args.limit_data
            },
            "training_config": {
                "effective_batch_size": per_device_batch_size * gradient_accumulation_steps,
                "total_steps": total_steps,
                "num_gpus": num_gpus,
                "learning_rate": training_args.learning_rate
            },
            "results": {
                "before_training": before_metrics,
                "after_training": after_metrics,
                "improvement": {
                    "exact_match_f1": after_metrics['exact_match_f1'] - before_metrics['exact_match_f1'],
                    "semantic_f1": after_metrics['semantic_f1'] - before_metrics['semantic_f1'],
                },
                "training_time": {
                    "actual_seconds": actual_training_time,
                    "time_per_step": actual_training_time / total_steps if total_steps > 0 else 0
                }
            }
        }
        
        with open(f"{model_dir}/results/results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {model_dir}/results/results.json")
        print(f"TensorBoard logs: {model_dir}/tensorboard_logs")
        print(f"\nTo view training progress:")
        print(f" tensorboard --logdir {model_dir}/tensorboard_logs")
        print(f" Then open: http://localhost:6006")
        print(f"\nOr view all experiments:")
        print(f" tensorboard --logdir ./experiments")
        


if __name__ == "__main__":
    main()