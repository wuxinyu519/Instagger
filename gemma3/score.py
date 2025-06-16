import pickle
import pandas as pd
import os
import json
def calculate_total_score(prediction_file, score_dataset_file):
    with open(prediction_file, 'rb') as f:
        prediction_data = pickle.load(f)

    with open(score_dataset_file, 'rb') as f:
        score_data = pickle.load(f)

    if isinstance(score_data, pd.DataFrame):
        score_df = score_data
    else:
        score_df = pd.DataFrame(score_data)

    print(f"Processing {len(prediction_data)} predictions...")
    print(f"Score dataset shape: {score_df.shape}")
    print(f"Score dataset columns: {list(score_df.columns)}")

    total_predict_score = 0.0
    total_maximum_score = 0.0
    valid_predictions = 0
    missing_samples = []
    error_samples = []

    for idx, sample in enumerate(prediction_data):
        sample_id = sample.get('sample_id', idx)
        predicted_model = sample.get('prediction', '').strip()

        matching_rows = score_df[score_df['sample_id'] == sample_id]
        if matching_rows.empty:
            missing_samples.append(sample_id)
            continue

        score_row = matching_rows.iloc[0]

        # Always compute maximum_score (from columns 2 to 10, i.e., 3rd to 11th columns)
        fixed_columns = list(score_df.columns[3:14])
        max_vals = []
        for col in fixed_columns:
            try:
                val = score_row[col]
                max_vals.append(float(val) if not pd.isna(val) else 0.0)
            except:
                max_vals.append(0.0)
        maximum_score = max(max_vals) if max_vals else 0.0  
        total_maximum_score += maximum_score  

        # Only compute predict_score if prediction exists
        predict_score = 0.0
        if predicted_model:
            for col in score_df.columns:
                if col.lower() == predicted_model.lower():
                    try:
                        val = score_row[col]
                        predict_score = float(val) if not pd.isna(val) else 0.0
                    except:
                        predict_score = 0.0
                    break

        total_predict_score += predict_score
        valid_predictions += 1

        if idx < 3:
            print(f"Sample {sample_id}: predict_score={predict_score}, max_score={maximum_score}")

    avg_predict_score = total_predict_score / valid_predictions if valid_predictions else 0.0
    avg_maximum_score = total_maximum_score / len(prediction_data) if prediction_data else 0.0
    score_ratio = total_predict_score / total_maximum_score if total_maximum_score else 0.0
    coverage = valid_predictions / len(prediction_data) if prediction_data else 0.0

    print(f"\nEvaluation Summary:")
    print(f"Total Predict Score:  {total_predict_score:.4f}")
    print(f"Total Maximum Score:  {total_maximum_score:.4f}")
    print(f"Predict / Maximum:    {score_ratio:.2%}")
    print(f"Avg Predict Score:    {avg_predict_score:.4f}")
    print(f"Avg Maximum Score:    {avg_maximum_score:.4f}")
    print(f"Coverage (predicted): {coverage:.2%}")
    print(f"Valid Predictions:    {valid_predictions}/{len(prediction_data)}")
    print(f"Missing Samples:      {len(missing_samples)}")
    print(f"Error Samples:        {len(error_samples)}")

    return {
        'total_predict_score': total_predict_score,
        'total_maximum_score': total_maximum_score,
        'avg_predict_score': avg_predict_score,
        'avg_maximum_score': avg_maximum_score,
        'score_ratio': score_ratio,
        'coverage': coverage,
        'valid_predictions': valid_predictions,
        'total_samples': len(prediction_data),
        'missing_samples': missing_samples,
        'error_samples': error_samples,
    }

def main():
    model_name = "gemma-3-4b-it"
    train_file = "data/train_samples.pkl"
    val_file = "data/val_samples.pkl"
    score_file = "../dataset/extractors/data/routerbench_0shot.pkl"
    save_path = f"./outputs/{model_name}/score_result.json"

    print("TRAINING SET")
    train_results = calculate_total_score(train_file, score_file)

    print("\nVALIDATION SET")
    val_results = calculate_total_score(val_file, score_file)

    print(f"\nFINAL SUMMARY")
    print(f"{'-'*40}")
    print(f"Train Predict Score: {train_results['total_predict_score']:.4f}")
    print(f"Train Max Score:     {train_results['total_maximum_score']:.4f}")
    print(f"Val Predict Score:   {val_results['total_predict_score']:.4f}")
    print(f"Val Max Score:       {val_results['total_maximum_score']:.4f}")
    print(f"Train Ratio:         {train_results['score_ratio']:.2%}")
    print(f"Val Ratio:           {val_results['score_ratio']:.2%}")
    print(f"Train Coverage:      {train_results['coverage']:.2%}")
    print(f"Val Coverage:        {val_results['coverage']:.2%}")
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, 'w') as f:
        json.dump({
            "train": train_results,
            "val": val_results
        }, f, indent=2)

    print(f"\n Evaluation results saved to {save_path}")
    return train_results, val_results

if __name__ == "__main__":
    train_results, val_results = main()
