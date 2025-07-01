#!/bin/bash
#PBS -N instagger_inference
#PBS -o inference.out
#PBS -e inference.err
#PBS -l walltime=08:00:00
#PBS -l nodes=1:gpu008
#PBS -q poderoso

cd $PBS_O_WORKDIR

# 环境
module load cuda12.6/toolkit
eval "$(conda shell.bash hook)"
conda activate nec



# GPU
export CUDA_VISIBLE_DEVICES=6,7
export TOKENIZERS_PARALLELISM=false

echo "Starting inference at $(date)"
nvidia-smi

# 运行推理（自动找最新checkpoint）
python inference.py \
    --data_path ../dataset/extractors/data/pii_prompts_val.pkl \
    --training_dir ./instagger-finetuned \
    --output_file pii_val_results.pkl \
    --num_samples 200 

echo "Inference completed at $(date)"