#!/bin/bash
#PBS -o full_eval.out
#PBS -e full_eval.err
#PBS -l walltime=24:00:00
#PBS -l nodes=1:gpu008
#PBS -q poderoso

cd $PBS_O_WORKDIR

# 环境
module load cuda12.6/toolkit
eval "$(conda shell.bash hook)"
conda activate nec

# GPU
export CUDA_VISIBLE_DEVICES=1,2
export TOKENIZERS_PARALLELISM=false

echo "Starting full evaluation at $(date)"
nvidia-smi


python inference_helmet.py \
    --checkpoint_path ./experiments/google_gemma-3-4b-it_20250730_221314/final_model/ \
    --data_dir ../HELMET/data/ruler/cwe/ \
    --output_prefix results_HELMET_fix \
    --save_raw_text \
    --batch_size 200 \
    --tensor_parallel_size 2 \
    --save_interval 500 \
    # --num_samples 1

echo "Full evaluation completed at $(date)"