#!/bin/bash
#PBS -N instagger_finetune
#PBS -o instagger.out
#PBS -e instagger.err
#PBS -l walltime=12:00:00
#PBS -l nodes=1:gpu008
#PBS -q poderoso

# 进入工作目录
cd $PBS_O_WORKDIR

# 定义数据路径
DATA_PATH="../dataset/extractors/clean/pii_tagged_total_clean.pkl"

# 加载必要的模块
module load cuda12.6/toolkit
eval "$(conda shell.bash hook)"
conda activate nec

# 设置环境变量
export CUDA_VISIBLE_DEVICES=1,2,3  # 3个GPU
export TOKENIZERS_PARALLELISM=false

# 打印GPU信息
echo "Available GPUs:"
nvidia-smi

# 运行训练
python train.py \
    --data_path $DATA_PATH \
    --output_dir ./instagger-finetuned \
    --batch_size 18 \
    --epochs 1 

echo "Training completed at $(date)"