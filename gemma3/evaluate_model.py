import os
import shutil
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from dataset_loader import load_dataset
from tqdm import tqdm
from collections import Counter
import numpy as np
from datetime import datetime

name = "router_selector_4b"
timestamp = datetime.now().strftime("%m%d_%H%M")
model_tag = f"{name}_{timestamp}"

def merge_lora_model():
    base_model_name = 'google/gemma-3-4b-it'
    lora_checkpoint_path = f'./outputs/{name}/checkpoint-29242'
    merged_model_path = f'./outputs/{model_tag}_merged'

    if not os.path.exists(lora_checkpoint_path):
        raise FileNotFoundError(f"LoRA checkpoint not found at {lora_checkpoint_path}")

    if os.path.exists(merged_model_path):
        shutil.rmtree(merged_model_path)

    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

        peft_model = PeftModel.from_pretrained(base_model, lora_checkpoint_path)
        merged_model = peft_model.merge_and_unload()

        os.makedirs(merged_model_path, exist_ok=True)
        merged_model.save_pretrained(merged_model_path, safe_serialization=False, max_shard_size="5GB")

        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        tokenizer.save_pretrained(merged_model_path)

        try:
            from transformers import AutoProcessor
            processor = AutoProcessor.from_pretrained(base_model_name)
            processor.save_pretrained(merged_model_path)
        except Exception as e:
            print(f"Warning: processor config not saved due to: {e}")

        del base_model, peft_model, merged_model
        torch.cuda.empty_cache()

        return merged_model_path
    except Exception as e:
        raise

def evaluate_dataset(llm, tokenizer, sampling_params, data, dataset_name):
    def preprocess_for_inference(example):
        CANDIDATE_MODELS = [
            "WizardLM/WizardLM-13B-V1.2", "claude-instant-v1", "claude-v1", "claude-v2",
            "gpt-3.5-turbo-1106", "gpt-4-1106-preview",
            "meta/code-llama-instruct-34b-chat", "meta/llama-2-70b-chat",
            "mistralai/mistral-7b-chat", "mistralai/mixtral-8x7b-chat",
            "no_model_correct", "zero-one-ai/Yi-34B-Chat"
        ]
        candidates_str = "\n".join([f"- {model}" for model in CANDIDATE_MODELS])
        return f"""Request: {example['prompt']}\n\nTask: {example['eval_name']}\nTags: {', '.join(example['tags'])}\n\nAvailable models:\n{candidates_str}\n\nSelect the best model to solve:"""

    prompts = [preprocess_for_inference(ex) for ex in data]
    expected_outputs = [ex['oracle_model_to_route_to'] for ex in data]
    outputs = llm.generate(prompts, sampling_params)

    correct = 0
    total_score = 0.0
    detailed_results = []

    for output, expected in zip(outputs, expected_outputs):
        generated = output.outputs[0].text.strip()
        prob_score = 0.0
        if hasattr(output.outputs[0], 'logprobs') and output.outputs[0].logprobs:
            token_logprobs = []
            for token_logprob_dict in output.outputs[0].logprobs:
                if isinstance(token_logprob_dict, dict):
                    logprob_values = [getattr(v, 'logprob', v) for v in token_logprob_dict.values()]
                    if logprob_values:
                        token_logprobs.append(max(logprob_values))
            if token_logprobs:
                sum_logprob = np.sum(token_logprobs)
                prob_score = min(np.exp(sum_logprob), 1.0)

        is_correct = expected.strip().lower() in generated.strip().lower().split()
        if is_correct:
            correct += 1
        total_score += prob_score

        detailed_results.append({
            'expected': expected, 'generated': generated,
            'correct': is_correct, 'prob_score': prob_score
        })

    total = len(data)
    return {
        'dataset_name': dataset_name,
        'total_samples': total,
        'correct_predictions': correct,
        'accuracy': correct / total,
        'total_score': total_score,
        'average_score': total_score / total,
        'theoretical_max_score': total * 1.0,
        'score_ratio_theoretical': total_score / total,
        'truncated_count': sum(len(tokenizer(p)["input_ids"]) > 2048 for p in prompts),
        'truncated_ratio': sum(len(tokenizer(p)["input_ids"]) > 2048 for p in prompts) / total,
        'avg_token_length': np.mean([len(tokenizer(p)["input_ids"]) for p in prompts]),
        'prediction_distribution': dict(Counter([r['generated'] for r in detailed_results])),
        'expected_distribution': dict(Counter(expected_outputs)),
        'failed_cases': [r for r in detailed_results if not r['correct']][:5],
        'detailed_results': detailed_results
    }

def evaluate_with_vllm(merged_model_path):
    llm = LLM(
        model=merged_model_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.8,
        max_model_len=2048,
        dtype="float16"
    )

    tokenizer = AutoTokenizer.from_pretrained(merged_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    sampling_params = SamplingParams(
        max_tokens=20,
        temperature=0.0,
        top_p=1.0,
        stop=["\n", "</s>", tokenizer.eos_token],
        skip_special_tokens=True,
        logprobs=5
    )

    train_data = load_dataset('data/train_samples.pkl')
    val_data = load_dataset('data/val_samples.pkl')

    train_results = evaluate_dataset(llm, tokenizer, sampling_params, train_data, "Train")
    val_results = evaluate_dataset(llm, tokenizer, sampling_params, val_data, "Validation")

    acc_diff = train_results['accuracy'] - val_results['accuracy']
    score_diff = train_results['average_score'] - val_results['average_score']

    result_path = f"{model_tag}_evaluation_results.txt"
    with open(result_path, "w", encoding="utf-8") as f:
        f.write("EVALUATION RESULTS SUMMARY\n")
        f.write(f"Train Accuracy: {train_results['accuracy']:.2%}\n")
        f.write(f"Validation Accuracy: {val_results['accuracy']:.2%}\n")
        f.write(f"Train Total Score: {train_results['total_score']:.4f}/{train_results['theoretical_max_score']:.0f}\n")
        f.write(f"Val Total Score: {val_results['total_score']:.4f}/{val_results['theoretical_max_score']:.0f}\n")
        f.write(f"Train Average Score: {train_results['average_score']:.4f}\n")
        f.write(f"Val Average Score: {val_results['average_score']:.4f}\n")
        f.write(f"Accuracy Difference: {acc_diff:.2%}\n")
        f.write(f"Score Difference: {score_diff:+.4f}\n")

    return {
        'train_accuracy': train_results['accuracy'],
        'val_accuracy': val_results['accuracy'],
        'train_total_score': train_results['total_score'],
        'val_total_score': val_results['total_score'],
        'train_average_score': train_results['average_score'],
        'val_average_score': val_results['average_score'],
        'train_results': train_results,
        'val_results': val_results
    }

def main():
    merged_model_path = f'./outputs/{model_tag}_merged'

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
        use_existing = input("Use existing merged model? (y/n): ").lower().strip()
        if use_existing != 'y':
            merged_model_path = merge_lora_model()
    else:
        merged_model_path = merge_lora_model()

    if not is_merged_model_valid(merged_model_path):
        raise RuntimeError(f"Failed to create valid merged model at {merged_model_path}")

    results = evaluate_with_vllm(merged_model_path)

    print(f"Train Accuracy: {results['train_accuracy']:.2%}")
    print(f"Validation Accuracy: {results['val_accuracy']:.2%}")
    print(f"Train Total Score: {results['train_total_score']:.4f}/{results['train_results']['theoretical_max_score']:.0f}")
    print(f"Val Total Score: {results['val_total_score']:.4f}/{results['val_results']['theoretical_max_score']:.0f}")
    print(f"Train Average Score: {results['train_average_score']:.4f}")
    print(f"Val Average Score: {results['val_average_score']:.4f}")

    return results

if __name__ == "__main__":
    results = main()