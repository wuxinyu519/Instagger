#!/bin/bash

# 配置
OPENAI_API_KEY="You_api_key"
MODEL_NAME="gpt-4o"
DATA_PATH="../dataset/extractors/data/pii_prompts.pkl"
OUTPUT_DIR="./chatgpt_inference_pii"
PYTHON_SCRIPT="gpt_infer.py"
LIMIT="100"

export OPENAI_API_KEY=$OPENAI_API_KEY

# 激活环境
eval "$(conda shell.bash hook)"
conda activate nec

# 创建输出目录
mkdir -p $OUTPUT_DIR

# 运行推理
echo "Starting ChatGPT inference..."
echo "Time: $(date)"

python3 $PYTHON_SCRIPT \
    --model_name $MODEL_NAME \
    --api_key $OPENAI_API_KEY \
    --data_path $DATA_PATH \
    --output_file $OUTPUT_DIR/chatgpt_results.pkl \
    --limit $LIMIT

echo "Completed at: $(date)"
echo "Results saved to: $OUTPUT_DIR/"