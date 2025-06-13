import os
import torch

import argparse
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, DataCollatorForSeq2Seq
)
from peft import get_peft_model, LoraConfig, TaskType

from dataset_loader import load_dataset
from evaluate import compute_accuracy
# Load model and apply LoRA
from transformers import BitsAndBytesConfig
os.environ["CUDA_VISIBLE_DEVICES"] = "0"




train_data = load_dataset('data/train_samples.pkl')
val_data = load_dataset('data/val_samples.pkl') if os.path.exists('data/val_samples.pkl') else []

# 合并训练集和验证集
all_data = train_data + val_data
CANDIDATE_MODELS = sorted(list(set(ex['oracle_model_to_route_to'] for ex in all_data)))
print(f"CANDIDATE_MODELS ({len(CANDIDATE_MODELS)}): {CANDIDATE_MODELS}")


#check labels
from collections import Counter
train_data = load_dataset('data/train_samples.pkl')
labels = [ex['oracle_model_to_route_to'] for ex in train_data]
label_dist = Counter(labels)

print("label distribution:")
for label, count in label_dist.most_common():
    print(f"{label}: {count} ({count/len(labels)*100:.1f}%)")


invalid_labels = [l for l in labels if l not in CANDIDATE_MODELS]
print(f"\ninvalid: {len(invalid_labels)}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='google/gemma-3-1b-it')
    parser.add_argument('--train_data_path', default='data/train_samples.pkl')
    parser.add_argument('--val_data_path', default='data/val_samples.pkl')
    parser.add_argument('--output_dir', default='./outputs/router_selector_1b_lr/')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--max_len', type=int, default=2048)
    parser.add_argument('--use_validation', action='store_true', help='Use validation set during training')
    return parser.parse_args()

def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Load train data
    print(f"Loading training data from {args.train_data_path}...")
    train_dataset = load_dataset(args.train_data_path)
    print(f"Training samples: {len(train_dataset)}")
    
    # Load validation data if using validation
    val_dataset = None
    if args.use_validation:
        print(f"Loading validation data from {args.val_data_path}...")
        val_dataset = load_dataset(args.val_data_path)
        print(f"Validation samples: {len(val_dataset)}")
    
    # Combine datasets to get all labels
    all_data = train_dataset + (val_dataset if val_dataset else [])
    all_labels = list(sorted(set(example["oracle_model_to_route_to"] for example in all_data)))
    label2id = {l: i for i, l in enumerate(all_labels)}
    id2label = {i: l for l, i in label2id.items()}
    
    print(f"Found {len(all_labels)} unique labels: {all_labels}")

    def preprocess(example):
        
        candidates_str = "\n".join([f"- {model}" for model in CANDIDATE_MODELS])
        
        full_prompt = f"""You are an AI model selector.

            Given the following request, task type, and tags, choose the **most suitable** model from the list of available candidates.

            Request:
            {example['prompt']}

            Task:
            {example['eval_name']}

            Tags:
            {', '.join(example['tags'])}

            Available Models:
            {candidates_str}

            Your answer should be the **name of the most appropriate model** from the above list."""

        
        target = example['oracle_model_to_route_to']
        
        
        if target not in CANDIDATE_MODELS:
            print(f"Warning: Skipping invalid label '{target}'")
            return None
        
        #  contraol lenth in total 2048
        prompt_tokens = tokenizer(full_prompt, truncation=True, max_length=args.max_len-20)["input_ids"]
        target_tokens = tokenizer(f" {target}", add_special_tokens=False)["input_ids"]
        
        
        input_ids = prompt_tokens + target_tokens
        if len(input_ids) > args.max_len:
            input_ids = input_ids[:args.max_len]
        
        
        labels = [-100] * len(prompt_tokens) + target_tokens
        if len(labels) > args.max_len:
            labels = labels[:args.max_len]
        
        # padding
        input_ids += [tokenizer.pad_token_id] * (args.max_len - len(input_ids))
        labels += [-100] * (args.max_len - len(labels))
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": [1 if token != tokenizer.pad_token_id else 0 for token in input_ids]
        }

    
    print("Preprocessing training data...")
    train_tokenized = [preprocess(ex) for ex in train_dataset]
    train_tokenized = [ex for ex in train_tokenized if ex is not None]
    print(f"Valid training samples: {len(train_tokenized)}")
    
    train_dataset_hf = Dataset.from_list(train_tokenized)
    
    
    val_dataset_hf = None
    if args.use_validation and val_dataset:
        print("Preprocessing validation data...")
        val_tokenized = [preprocess(ex) for ex in val_dataset]
        val_tokenized = [ex for ex in val_tokenized if ex is not None]
        print(f"Valid validation samples: {len(val_tokenized)}")
        val_dataset_hf = Dataset.from_list(val_tokenized)

    

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map={"": 0},
        torch_dtype=torch.float16,
        offload_folder="./offload",
        attn_implementation="eager",
        low_cpu_mem_usage=True
    )

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    model = get_peft_model(model, peft_config)
    
    
    model.print_trainable_parameters()

   
    training_args = TrainingArguments(
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1 if args.use_validation else args.batch_size,
        num_train_epochs=args.epochs,
        eval_strategy="epoch" if args.use_validation else "no",
        save_strategy="epoch",
        output_dir=args.output_dir,
        logging_dir=os.path.join(args.output_dir, "logs"),
        learning_rate=args.lr,
        logging_steps=10,
        save_total_limit=3,
        fp16=True,
        max_grad_norm=1.0,
        load_best_model_at_end=args.use_validation,
        metric_for_best_model="eval_loss" if args.use_validation else None,
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        report_to=None,
    )

    trainer_kwargs = {
        "model": model,
        "tokenizer": tokenizer,
        "args": training_args,
        "train_dataset": train_dataset_hf,
        "data_collator": DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt"),
    }
    
    
    if args.use_validation and val_dataset_hf:
        trainer_kwargs["eval_dataset"] = val_dataset_hf
        trainer_kwargs["compute_metrics"] = lambda p: compute_accuracy(p, tokenizer, id2label)

    trainer = Trainer(**trainer_kwargs)

   
    import json
    config = {
        "model_name": args.model_name,
        "train_data_path": args.train_data_path,
        "val_data_path": args.val_data_path,
        "candidate_models": CANDIDATE_MODELS,
        "training_samples": len(train_tokenized),
        "validation_samples": len(val_tokenized) if val_dataset_hf else 0,
        "use_validation": args.use_validation,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "max_length": args.max_len
    }
    
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "training_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print("Starting training...")
    trainer.train()
    
    print("Saving final model...")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    
    print("Training completed!")

if __name__ == "__main__":
    main()