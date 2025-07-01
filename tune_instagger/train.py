#!/usr/bin/env python3
import os
import pickle
import json
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import argparse

class InsTaggerTrainer:
    def __init__(self, model_name: str = "OFA-Sys/InsTagger"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            # device_map="auto",
            trust_remote_code=True
        )
        
        
  
        self.model.generation_config.temperature = None
        self.model.generation_config.top_p = None

    def load_data(self, data_path: str, limit: int = None):
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
  
        if limit:
            data = data[:limit]
        
        training_data = []
        template = "You are a helpful assistant. Please identify tags of user intentions in the following user query. Query: {query} Assistant: {tags}"
        
        for item in data:
            if isinstance(item, dict):
                
                prompt = item.get('prompt')
                tags = item.get('tags')
                
                if prompt and tags:
                    tags_str = json.dumps(tags, ensure_ascii=False) if isinstance(tags, list) else str(tags)
                    formatted_text = template.format(query=prompt, tags=tags_str)
                    training_data.append({"text": formatted_text})
        
        print(f"Loaded {len(training_data)} samples")
        return training_data

    def tokenize_function(self, examples):
        try:
            tokens = self.tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",  # padding到batch内最大长度
                max_length=1800
                # return_tensors=None,
            )
            tokens["labels"] = tokens["input_ids"].copy()
            return tokens
        except Exception as e:
           
            print(f"Tokenization error: {e}")
            print(f"Batch size: {len(examples['text'])}")
            for i, text in enumerate(examples["text"]):
                print(f"Sample {i} length: {len(text) if text else 0}, type: {type(text)}")
                if text:
                    print(f"Sample {i} preview: {text[:100]}...")
            raise e
    def train(self, data_path: str, output_dir: str = "./instagger-finetuned", 
              batch_size: int = 512, epochs: int = 1, limit: int = None):
        
        # 加载数据
        training_data = self.load_data(data_path, limit=limit)
        dataset = Dataset.from_list(training_data)
        
        # 分词
        tokenized_dataset = dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
        )
        
        # 计算每设备batch size
        num_gpus = torch.cuda.device_count()
        per_device_batch_size = batch_size // num_gpus if num_gpus > 0 else batch_size
        
        # 训练参数
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=per_device_batch_size,
            learning_rate=2e-5,
            logging_steps=100, 
            save_steps=1000,
            logging_dir=f"{output_dir}/logs",  
            save_total_limit=3,
            bf16=True,
            remove_unused_columns=False,
            report_to="tensorboard",  # 使用tensorboard记录
            logging_first_step=True,  
            save_strategy="steps",
            evaluation_strategy="no",
            dataloader_drop_last=False,
            load_best_model_at_end=False,
        )
        
        # 训练
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer, mlm=False
            ),
        )
        
        
        # if hasattr(trainer.model, "generation_config"):
        #     trainer.generation_config.temperature = None
        #     trainer.generation_config.top_p = None

        
        trainer.train()

        final_model_dir = os.path.join(output_dir, "final_model")
        os.makedirs(final_model_dir, exist_ok=True)
        trainer.save_model(final_model_dir)
        self.tokenizer.save_pretrained(final_model_dir)
        
        # 保存训练日志
        if hasattr(trainer.state, 'log_history'):
            import json
            with open(f"{output_dir}/training_log.json", "w") as f:
                json.dump(trainer.state.log_history, f, indent=2)
        
        print(f"Training completed! Model saved to {output_dir}/final_model")
        print(f"View training logs: tensorboard --logdir {output_dir}/logs")


        # 测试训练数据的第一个样本
        if training_data:
            print("\n" + "="*60)
            print("Testing trained model with first training sample:")
            print("="*60)
            
            # 从第一个训练样本中提取原始query
            first_sample = training_data[0]["text"]
            # 从formatted text中提取query部分
            query_start = first_sample.find("Query: ") + 7
            query_end = first_sample.find(" Assistant:")
            if query_start > 6 and query_end > query_start:
                original_query = first_sample[query_start:query_end]
                
                try:
                    result = self.inference(original_query)
                    print(f"Original Query: {original_query}")
                    print(f"Generated Tags: {result}")
                except Exception as e:
                    print(f"Error in inference: {e}")
            else:
                print("Could not extract query from training sample")
            
            print("="*60)

    def inference(self, query: str):
        prompt = f"You are a helpful assistant. Please identify tags of user intentions in the following user query. Query: {query} Assistant: "
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=inputs["input_ids"].shape[1] + 256,
                temperature=0.9,
                top_p= 0.6,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response[len(prompt):].strip()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./instagger-finetuned")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    
    args = parser.parse_args()
    
    trainer = InsTaggerTrainer()
    trainer.train(
        data_path=args.data_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        limit=args.limit
    )

if __name__ == "__main__":
    main()