#!/bin/bash
#PBS -N tune_with_pii
#PBS -o gemma_tune_with_pii.out
#PBS -e gemma_tune_with_pii.err
#PBS -l walltime=48:00:00
#PBS -l nodes=1:gpu008
#PBS -q poderoso

cd $PBS_O_WORKDIR

# PKL_PATH="../gemma3/gemma_infered_result/cleaned_tags_batch/individual_files/tagged_pii_gemma_results_full_cleaned.pkl"
PKL_PATH="../gemma3/gemma27b_infered_prompt2/cleaned_tags_batch/individual_files/general_data_train_cleaned.pkl"
module load cuda12.6/toolkit
eval "$(conda shell.bash hook)"
conda activate nec

export CUDA_VISIBLE_DEVICES=5,6  #1,2 or 3,4,5,6
export TOKENIZERS_PARALLELISM=false
# export HF_HOME="/tmp/hf_cache"

echo "Testing started at $(date)"
echo "Available GPUs:"
nvidia-smi

echo "Data path: $PKL_PATH"

# 直接用python运行 google/gemma-3-4b-it,OFA-Sys/InsTagger
python tune_with_pii.py \
    --pkl_path $PKL_PATH \
    --models "OFA-Sys/InsTagger" \
    --batch_size 6 \
    --use_all_data \
    # --limit_data 10000 \

echo "Testing completed at $(date)"