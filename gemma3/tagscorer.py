import pickle
import pandas as pd
import os
import json
import numpy as np
from collections import defaultdict, Counter
import math

class TagRouterScorer:
    def __init__(self):
        self.tag_score_mapping = {}  # (model, tag) -> score
        self.tag_frequencies = {}    # tag -> frequency
        self.total_tag_frequency = 0  # 所有标签频次的总和
        
    def build_tag_score_mapping(self, score_dataset, prediction_data):
        
        if isinstance(score_dataset, pd.DataFrame):
            score_df = score_dataset
        else:
            score_df = pd.DataFrame(score_dataset)
            
        # 获取模型列
        model_columns = list(score_df.columns)[3:14]
        
   
        all_tags = []
        sample_tags = {}
        
        for sample in prediction_data:
            sample_id = sample.get('sample_id')
            tags = sample.get('tags', [])
            if isinstance(tags, str):
                tags = [tags]
            sample_tags[sample_id] = tags
            all_tags.extend(tags)
        
        # 计算标签频次 (countt)
        self.tag_frequencies = Counter(all_tags)
        self.total_tag_frequency = sum(self.tag_frequencies.values())
        
        print(f"Total tags: {len(self.tag_frequencies)}, Total frequency: {self.total_tag_frequency}")
        
   
        tag_model_comparisons = defaultdict(lambda: defaultdict(lambda: {'win': 0, 'tie': 0, 'loss': 0}))
        
   
        score_lookup = {}
        for idx, score_row in score_df.iterrows():
            sample_id = score_row['sample_id']
            score_lookup[sample_id] = score_row
        
 
        for sample_id, tags in sample_tags.items():
            if sample_id not in score_lookup:
                continue
                
            score_row = score_lookup[sample_id]
            
      
            model_scores = {}
            for model in model_columns:
                try:
                    val = score_row[model]
                    if isinstance(val, list) or (isinstance(val, str) and len(val) > 10):
                        continue
                    score = float(val) if not pd.isna(val) else 0.0
                    model_scores[model] = score
                except (ValueError, TypeError):
                    model_scores[model] = 0.0
            
            if not model_scores:
                continue
        
            max_score = max(model_scores.values())
            
            # 为每个标签，计算每个模型相对于最优表现的比较结果
            for tag in tags:
                for model, score in model_scores.items():
                
                    if score == max_score and score > 0:
                      
                        tag_model_comparisons[tag][model]['win'] += 1
                    elif score == 0:
                      
                        tag_model_comparisons[tag][model]['loss'] += 1
                    else:
                    
                        tag_model_comparisons[tag][model]['tie'] += 1
        
        # 计算TagRouter的tag-score映射
        #  score(Mi, t) = wt · Σ count_t,Mi(r) · sr
        
        s_win = 1.0
        s_tie = 0.15  
        s_loss = -1.0
        
        for tag in self.tag_frequencies:
            # wt = (1 - exp(-countt)) / Σcountt')
            tag_freq = self.tag_frequencies[tag]
            w_t = (1 - math.exp(-tag_freq)) / self.total_tag_frequency
            
            for model in model_columns:
                if model in tag_model_comparisons[tag]:
                    comparisons = tag_model_comparisons[tag][model]
                    
                    
                    base_score = (comparisons['win'] * s_win + 
                                 comparisons['tie'] * s_tie + 
                                 comparisons['loss'] * s_loss)
                    
                    
                    final_score = w_t * base_score
                    
                    self.tag_score_mapping[(model, tag)] = final_score
                else:
                  
                    self.tag_score_mapping[(model, tag)] = 0.0
        
        print(f"Built tag-score mapping with {len(self.tag_score_mapping)} entries")
        
        
        print("Sample tag-score mappings:")
        sorted_mappings = sorted(self.tag_score_mapping.items(), key=lambda x: abs(x[1]), reverse=True)
        for (model, tag), score in sorted_mappings[:5]:
            comparisons = tag_model_comparisons[tag][model]
            print(f"  {model} + '{tag}': {score:.4f} (W:{comparisons['win']}, T:{comparisons['tie']}, L:{comparisons['loss']})")
    
    def select_best_model_by_tags(self, tags, available_models):
        """
        M*(q) = argmax_M∈M Σt∈T(q) score(M, t)
        """
        if isinstance(tags, str):
            tags = [tags]
            
        model_scores = {}
        for model in available_models:
            # 计算该模型在所有查询标签上的分数总和
            total_score = 0.0
            for tag in tags:
                tag_score = self.tag_score_mapping.get((model, tag), 0.0)
                total_score += tag_score
            model_scores[model] = total_score
        
        if model_scores:
            # 选择分数最高的模型
            best_model = max(model_scores, key=model_scores.get)
            return best_model
        else:
            return available_models[0] if available_models else None

def calculate_scores(prediction_file, score_dataset_file):

    with open(prediction_file, 'rb') as f:
        prediction_data = pickle.load(f)

    with open(score_dataset_file, 'rb') as f:
        score_data = pickle.load(f)

    if isinstance(score_data, pd.DataFrame):
        score_df = score_data
    else:
        score_df = pd.DataFrame(score_data)

    print(f"Processing {len(prediction_data)} predictions...")


    tag_scorer = TagRouterScorer()
    tag_scorer.build_tag_score_mapping(score_df, prediction_data)

    total_tagrouter_score = 0.0
    total_maximum_score = 0.0


    model_columns = list(score_df.columns)[3:14]

    for sample in prediction_data:
        sample_id = sample.get('sample_id')

    
        tags = sample.get('tags', [])
        if isinstance(tags, str):
            tags = [tags]

    
        matching_rows = score_df[score_df['sample_id'] == sample_id]
        if matching_rows.empty:
            continue

        score_row = matching_rows.iloc[0]

        # 计算Maximum Score
        maximum_score = 0.0
        for model_name in model_columns:
            try:
                val = score_row[model_name]
                if isinstance(val, list) or (isinstance(val, str) and len(val) > 10):
                    continue
                score = float(val) if not pd.isna(val) else 0.0
                maximum_score = max(maximum_score, score)
            except (ValueError, TypeError):
                continue
        
        total_maximum_score += maximum_score

        tagrouter_score = 0.0
        if tags:
            tagrouter_best_model = tag_scorer.select_best_model_by_tags(tags, model_columns)
            if tagrouter_best_model:
                try:
                    val = score_row[tagrouter_best_model]
                    tagrouter_score = float(val) if not pd.isna(val) else 0.0
                except:
                    tagrouter_score = 0.0
        
        total_tagrouter_score += tagrouter_score


    tagrouter_ratio = total_tagrouter_score / total_maximum_score if total_maximum_score else 0.0

    return {
        'total_maximum_score': round(total_maximum_score, 2),
        'total_tagrouter_score': round(total_tagrouter_score, 2),
        'tagrouter_ratio': round(tagrouter_ratio, 4)
    }

def main():
    model_name = "gemma-3-1b-it_multiclass"
    train_file = "data/train_samples_1b_multi.pkl"
    val_file = "data/val_samples_1b_multi.pkl"
    score_file = "../dataset/extractors/data/routerbench_0shot.pkl"
    save_path = f"./outputs/{model_name}/tagrouter_comparison.json"

    print("=== TagRouter Strict Implementation ===")
    print("Computing TagRouter comparison using exact paper formulas...")
    
    print("\nTRAINING SET:")
    train_results = calculate_scores(train_file, score_file)
    
    print("\nVALIDATION SET:")
    val_results = calculate_scores(val_file, score_file)

    print(f"\n=== FINAL RESULTS ===")
    print(f"TRAIN:")
    print(f"  Maximum Score: {train_results['total_maximum_score']}")
    print(f"  TagRouter Score: {train_results['total_tagrouter_score']}")
    print(f"  TagRouter Ratio: {train_results['tagrouter_ratio']:.2%}")
    
    print(f"VAL:")
    print(f"  Maximum Score: {val_results['total_maximum_score']}")
    print(f"  TagRouter Score: {val_results['total_tagrouter_score']}")
    print(f"  TagRouter Ratio: {val_results['tagrouter_ratio']:.2%}")
    

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump({
            "train": train_results,
            "val": val_results,
            "note": "Strict TagRouter implementation following paper formulas exactly"
        }, f, indent=2)

    print(f"\nResults saved to {save_path}")
    return train_results, val_results

if __name__ == "__main__":
    train_results, val_results = main()