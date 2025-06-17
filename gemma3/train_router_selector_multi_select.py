import os
import torch
import argparse
import json
import csv
from datetime import datetime
from collections import Counter
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, DataCollatorForSeq2Seq, 
    TrainerCallback, EarlyStoppingCallback  
)
from peft import get_peft_model, LoraConfig, TaskType
from dataset_loader import load_dataset
from evaluate import compute_accuracy


def extract_ground_truth(sample, start_col=3, end_col=14):
    """从样本中提取ground truth模型列表 - 基于分数"""
    # 获取所有列名
    if hasattr(sample, 'keys'):
        all_columns = list(sample.keys())
    elif hasattr(sample, 'index'):  # pandas Series
        all_columns = list(sample.index)
    else:
        return []
    
    # 选择索引3到14的列（模型名列）
    model_columns = all_columns[start_col:end_col+1] if len(all_columns) > end_col else all_columns[start_col:]
    
    # 收集模型名和对应分数
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
        # 这样在build_training_prompt中会设置为"no_model_correct"
    
    return ground_truth_models

# ---------------------------
# Prompt Builder 
# ---------------------------
class PromptBuilder:
    
    def __init__(self, candidate_models, max_prompt_length=1800):
        self.candidate_models = candidate_models
        self.candidates_str = ", ".join(sorted(candidate_models))
        self.max_prompt_length = max_prompt_length
    
    def _normalize_prompt(self, prompt):
        """标准化输入prompt"""
        if isinstance(prompt, list):
            return ' '.join(str(p) for p in prompt)
        elif not isinstance(prompt, str):
            return str(prompt)
        return prompt
    
    
    
    def build_training_prompt(self, example):
        user_prompt = self._normalize_prompt(example['prompt'])
        tags = example.get('tags', [])
        
        input_prompt = self._build_input_prompt(
            eval_name=example['eval_name'],
            user_prompt=user_prompt,
            tags=tags
        )
        
        ground_truth_models = extract_ground_truth(example)
        
        if not ground_truth_models:
            target = ' {"best_model": "no_model_correct"}'
        else:
            
            target_parts = []
            for model in ground_truth_models:
                target_parts.append(f'{{"best_model": "{model}"}}')
            target = ' ' + ' '.join(target_parts)
        
        return input_prompt, target

    def build_inference_prompt(self, eval_name, user_prompt, tags=None):
        
        user_prompt = self._normalize_prompt(user_prompt)
        return self._build_input_prompt(eval_name, user_prompt, tags)

    def _build_input_prompt(self, eval_name, user_prompt, tags=None):
       
        truncated_prompt = user_prompt[:self.max_prompt_length]
        tag_str = ", ".join(tags) if tags else "N/A"

    

        template = """You are a model selector. Choose the **best model(s)** from this list: {candidates}

        For the task: {eval_name}
        Tags: {tag_str}
        Question: {user_prompt}

        You can select multiple models if they are equally good. For each selected model, output a separate JSON object:
        {{"best_model": "model_name"}}
        If multiple models are good, output like: {{"best_model": "model1"}} {{"best_model": "model2"}}

        IMPORTANT: If NO model from the list is suitable for this task, output:
        {{"best_model": "no_model_correct"}}"""

        return template.format(
            candidates=self.candidates_str,
            eval_name=eval_name,
            tag_str=tag_str,
            user_prompt=truncated_prompt
        )
    def extract_prediction(self, raw_output):
       
        return self.extract_prediction_simple(raw_output)
    def extract_prediction_simple(self, raw_output):
        
        if not raw_output:
            return []
        
        import re
        
        
        pattern = r'"best_model"\s*:\s*"([^"]*)'  
        matches = re.findall(pattern, raw_output, re.IGNORECASE)
        
        models = []
        for match in matches:
            model = match.strip()
            if model:
                models.append(model)
        
        print(f"Extracted models: {models}")
        return models
    
    def is_prediction_correct(self, predictions, expected_models):
       
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

# ---------------------------
# Callback to log loss
# ---------------------------
class LossLoggingCallback(TrainerCallback):
    def __init__(self, output_path):
        self.output_path = output_path
  
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(self.output_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['step', 'epoch', 'loss', 'learning_rate', 'timestamp'])

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or 'loss' not in logs:
            return
        
        # 只在主进程记录
        if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            return
        
    
        loss_val = logs.get("loss")
        if loss_val is not None and (torch.isnan(torch.tensor(loss_val)) or torch.isinf(torch.tensor(loss_val))):
            print(f"⚠️ Warning: Abnormal loss detected: {loss_val}")
        
        with open(self.output_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                state.global_step,
                round(state.epoch, 4),
                loss_val,
                logs.get("learning_rate"),
                datetime.now().isoformat()
            ])

# ---------------------------
# Production prediction callback
# ---------------------------
class ProductionPredictionCallback(TrainerCallback):
    def __init__(self, tokenizer, prompt_builder, output_path, eval_samples, max_samples=5):
        self.tokenizer = tokenizer
        self.prompt_builder = prompt_builder
        self.output_path = output_path
        self.eval_samples = eval_samples[:max_samples]
        
        # 只在主进程创建文件
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(self.output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['epoch', 'sample_idx', 'eval_name', 'prediction', 'raw_output', 'expected', 'match'])

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        # 只在主进程执行预测
        if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            return
            
        print(f"\nPredictions at epoch {round(state.epoch, 2)}:")
        
        try:
            model.eval()
            correct = 0
            total = len(self.eval_samples)
            
            with open(self.output_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                for i, example in enumerate(self.eval_samples):
                    try:
                
                        full_prompt = self.prompt_builder.build_inference_prompt(
                            eval_name=example['eval_name'],
                            user_prompt=example['prompt']
                        )
                        
                     
                        inputs = self.tokenizer(
                            full_prompt,
                            return_tensors="pt",
                            truncation=True,
                            max_length=1800,  # 和训练时一致
                            padding=False
                        )
                        
               
                        device = next(model.parameters()).device
                        inputs = {k: v.to(device) for k, v in inputs.items()}
                        
                        
                        with torch.no_grad():
                            outputs = model.generate(
                                **inputs,
                                max_new_tokens=100,
                                do_sample=True,
                                pad_token_id=self.tokenizer.pad_token_id,
                                eos_token_id=self.tokenizer.eos_token_id,
                                use_cache=False
                            )
                        
                     
                        input_length = inputs["input_ids"].shape[1]
                        generated = outputs[0][input_length:]
                        raw_output = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
                        
                       
                        prediction = self.prompt_builder.extract_prediction(raw_output)
                       
                        expected_models = extract_ground_truth(example)
                        expected = expected_models[0] if expected_models else 'no_model_correct'
                        
                        
                        is_match = self.prompt_builder.is_prediction_correct(prediction, expected_models)
                        
                        if is_match:
                            correct += 1
                        
                        print(f"Sample {i+1}: {example['eval_name']}")
                        print(f"Raw Output: '{raw_output}'")
                        print(f"Extracted Prediction: '{prediction}'")
                        print(f"Expected: '{expected}'")
                        print(f"✅ Match: {is_match}\n")
                        
                        writer.writerow([
                            round(state.epoch, 2),
                            i,
                            example["eval_name"],
                            prediction,
                            raw_output.replace('\n', '\\n'),  # 转义换行符以便CSV保存
                            expected,
                            is_match
                        ])
                        
                    except Exception as e:
                        print(f"Error in sample {i+1}: {str(e)}")
                        writer.writerow([
                            round(state.epoch, 2),
                            i,
                            "ERROR",
                            f"Error: {str(e)[:30]}",
                            "ERROR",
                            "ERROR",
                            False
                        ])
                        continue
            
            accuracy = correct / total if total > 0 else 0
            print(f"Epoch {round(state.epoch, 2)} Accuracy: {accuracy:.1%} ({correct}/{total})")
                        
        except Exception as e:
            print(f"Error in prediction callback: {str(e)}")
        finally:
            model.train()

# ---------------------------
# Main training function
# ---------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Router Model Training")
    
    # Model and data
    parser.add_argument('--model_name', default='google/gemma-3-4b-it')
    parser.add_argument('--train_data_path', default='data/train_samples.pkl')
    parser.add_argument('--val_data_path', default='data/val_samples.pkl')
    parser.add_argument('--output_dir', default='./outputs/gemma-3-4b-it_multiclass/')
    
    # Training params
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size per device")
    parser.add_argument('--epochs', type=int, default=10, help="Number of epochs")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--max_len', type=int, default=1800, help="Max sequence length")
    
    # Data size (for testing vs production)
    parser.add_argument('--train_samples', type=int, default=-1, help="Number of train samples (-1 for all)")
    parser.add_argument('--val_samples', type=int, default=100, help="Number of validation samples")
    
    # Multi-GPU settings
    parser.add_argument('--gpu_ids', default='2,3', help="GPU IDs to use")
    
    # Validation and early stopping
    parser.add_argument('--use_validation', action='store_true', default=True)
    parser.add_argument('--early_stopping_patience', type=int, default=2, help="Early stopping patience")
    
    # LoRA settings
    parser.add_argument('--lora_r', type=int, default=8, help="LoRA rank")
    parser.add_argument('--lora_alpha', type=int, default=16, help="LoRA alpha")
    
    # Prompt settings
    parser.add_argument('--max_prompt_length', type=int, default=1800, help="Max prompt length")
    
    return parser.parse_args()

def setup_multi_gpu(gpu_ids_str):
    """设置多GPU环境"""
    if gpu_ids_str and ',' in gpu_ids_str:
        gpu_ids = [int(x.strip()) for x in gpu_ids_str.split(',')]
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids_str
        print(f"🔧 Multi-GPU setup: Using GPUs {gpu_ids}")
        return len(gpu_ids)
    else:
        gpu_id = int(gpu_ids_str) if gpu_ids_str else 0
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        print(f"🔧 Single-GPU setup: Using GPU {gpu_id}")
        return 1

def main():
    args = parse_args()
    num_gpus = setup_multi_gpu(args.gpu_ids)
    
 
    torch.manual_seed(42)
    
    print(f"   Starting Production Training")
    print(f"   Model: {args.model_name}")
    print(f"   Output: {args.output_dir}")
    print(f"   GPUs: {num_gpus}")
    print(f"   Batch size per GPU: {args.batch_size}")
    print(f"   Total effective batch size: {args.batch_size * num_gpus}")
    print(f"   Learning rate: {args.lr}")
    print(f"   Max epochs: {args.epochs}")
    print(f"   Early stopping patience: {args.early_stopping_patience}")
    
  
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Tokenizer: vocab_size={len(tokenizer)}, pad_token_id={tokenizer.pad_token_id}")


    print("Loading data...")
    train_data = load_dataset(args.train_data_path)
    if args.train_samples > 0:
        train_data = train_data[:args.train_samples]
        print(f"   Using {len(train_data)} training samples")
    else:
        print(f"   Using all {len(train_data)} training samples")
    
    val_data = []
    if args.use_validation:
        val_data = load_dataset(args.val_data_path)[:args.val_samples]
        print(f"   Using {len(val_data)} validation samples")
    
    
    if train_data:
        sample = train_data[0]
        if hasattr(sample, 'keys'):
            all_columns = list(sample.keys())
        elif hasattr(sample, 'index'):
            all_columns = list(sample.index)
        else:
            all_columns = []
        CANDIDATE_MODELS = all_columns[3:15]
    else:
        CANDIDATE_MODELS = []
    print(f"   Found {len(CANDIDATE_MODELS)} candidate models")
        
    
    prompt_builder = PromptBuilder(
        candidate_models=CANDIDATE_MODELS,
        max_prompt_length=args.max_prompt_length
    )
    print(f"Prompt builder initialized with {len(CANDIDATE_MODELS)} models")

    def preprocess_for_training(example):
      
        try:
            full_prompt, target = prompt_builder.build_training_prompt(example)

            prompt_encoding = tokenizer(
                full_prompt,
                truncation=True,
                max_length=args.max_len - 50,
                add_special_tokens=True
            )
            
            target_encoding = tokenizer(
                target,
                add_special_tokens=False
            )
            
            prompt_tokens = prompt_encoding["input_ids"]
            target_tokens = target_encoding["input_ids"]
            
            if not target_tokens:
                return None
            
            # 合并
            input_ids = prompt_tokens + target_tokens
            labels = [-100] * len(prompt_tokens) + target_tokens
            
            # 截断
            input_ids = input_ids[:args.max_len]
            labels = labels[:args.max_len]
            
   
            pad_length = args.max_len - len(input_ids)
            if pad_length > 0:
                input_ids += [tokenizer.pad_token_id] * pad_length
                labels += [-100] * pad_length
            
            attention_mask = [1 if token_id != tokenizer.pad_token_id else 0 for token_id in input_ids]
            

            valid_labels = [l for l in labels if l != -100]
            if not valid_labels:
                return None
            
            return {
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": attention_mask
            }
            
        except Exception as e:
            print(f"Failed to preprocess example: {str(e)}")
            return None

    print("Preprocessing data...")
    train_processed = [preprocess_for_training(ex) for ex in train_data]
    train_processed = [x for x in train_processed if x is not None]
    train_dataset = Dataset.from_list(train_processed)
    
    val_dataset = None
    val_processed = []
    if args.use_validation and val_data:
        val_processed = [preprocess_for_training(ex) for ex in val_data]
        val_processed = [x for x in val_processed if x is not None]
        val_dataset = Dataset.from_list(val_processed)
    
    print(f"Processed {len(train_processed)} training, {len(val_processed)} validation samples")

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        torch_dtype=torch.float32,
        attn_implementation="eager"
    )

    model_vocab_size = getattr(model.config, 'vocab_size', model.get_input_embeddings().weight.shape[0])
    tokenizer_vocab_size = len(tokenizer)

    print(f"Model vocab: {model_vocab_size}, Tokenizer vocab: {tokenizer_vocab_size}")

    if tokenizer_vocab_size != model_vocab_size:
        model.resize_token_embeddings(tokenizer_vocab_size)
        print(f"Resized embeddings: {model_vocab_size} → {tokenizer_vocab_size}")
    

    model.config.use_cache = False
    if hasattr(model, 'generation_config'):
        model.generation_config.use_cache = False

    print("Applying LoRA...")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]  # 更多模块用于生产
    )
    model = get_peft_model(model, peft_config)
    
    # 统计参数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🎯 Trainable: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.2f}%)")


    training_args = TrainingArguments(
 
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        
  
        output_dir=args.output_dir,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=20,
        save_strategy="epoch",
        save_total_limit=5,
        

        eval_strategy="epoch" if args.use_validation else "no",
        load_best_model_at_end=args.use_validation,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        

        fp16=True,
        dataloader_num_workers=4 if num_gpus > 1 else 2,
        dataloader_pin_memory=True,

        max_grad_norm=1.0,
        warmup_ratio=0.01,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        
 
        ddp_find_unused_parameters=False,
        
        
        report_to=None,
        remove_unused_columns=True,
        prediction_loss_only=True,
        include_inputs_for_metrics=False,
        label_names=["labels"]
    )


    callbacks = [
        LossLoggingCallback(os.path.join(args.output_dir, "training_loss.csv"))
    ]
    
    # 添加早停
    if args.use_validation and val_dataset:
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=args.early_stopping_patience
        ))
        print(f"Early stopping enabled with patience {args.early_stopping_patience}")
    
    # 添加预测监控 - 使用统一的prompt构造器
    if val_data:
        callbacks.append(ProductionPredictionCallback(
            tokenizer=tokenizer,
            prompt_builder=prompt_builder,
            output_path=os.path.join(args.output_dir, "predictions.csv"),
            eval_samples=val_data,
            max_samples=5
        ))

    # 创建trainer
    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer,
            pad_to_multiple_of=8,
            return_tensors="pt",
            padding=True
        ),
        callbacks=callbacks
    )

    # 开始训练
    print("\nStarting training...")
    try:
        trainer.train()
        trainer.save_model()
        tokenizer.save_pretrained(args.output_dir)
        
   
        prompt_config = {
            'candidate_models': CANDIDATE_MODELS,
            'max_prompt_length': args.max_prompt_length,
            'prompt_template': prompt_builder._build_input_prompt('example_task', 'example_prompt')
        }
        prompt_config_path = os.path.join(args.output_dir, "prompt_config.json")
        with open(prompt_config_path, 'w') as f:
            json.dump(prompt_config, f, indent=2)
        print(f"Prompt config saved to {prompt_config_path}")
        
        print("Training completed successfully!")
        

        config_path = os.path.join(args.output_dir, "training_config.json")
        with open(config_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
        print(f"Training config saved to {config_path}")
        
    except Exception as e:
        print(f"Training failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()