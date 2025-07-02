#!/bin/bash
#PBS -N gemma_inference
#PBS -q poderoso
#PBS -l nodes=1:gpu008
#PBS -l walltime=24:00:00
#PBS -e gemma_inference.err
#PBS -o gemma_inference.log
cd $PBS_O_WORKDIR


# =====================================================
# Configuration Section - Modify these paths
# =====================================================
export CUDA_VISIBLE_DEVICES=1,2,3,4
# export NCCL_TIMEOUT=1800
# export NCCL_IB_DISABLE=1  
# Model and data configuration
MODEL_NAME="google/gemma-3-27b-it"
DATA_PATH="../dataset/extractors/clean/pii_tagged_total_clean.pkl"
OUTPUT_DIR="./pii_inference"
# OUTPUT_DIR="./pii_inference_with_changed_prompt"
PYTHON_SCRIPT="gemma3_27b.py"

# Training parameters
BATCH_SIZE=4
TENSOR_PARALLEL_SIZE=4  # Should match number of GPUs
GPU_MEMORY_UTIL=0.85
CHECKPOINT_INTERVAL=5   # Save checkpoint every 5 batches
LIMIT="10"  # Leave empty to process all data, or set a number like "1000"

# =====================================================
# Environment Setup - Modify according to your cluster
# =====================================================
# Load required modules (adjust according to your cluster)
module load cuda12.6/toolkit
eval "$(conda shell.bash hook)"
conda activate nec

# =====================================================
# Job Execution
# =====================================================

# Print job information
echo "======================================================="
echo "JOB INFORMATION"
echo "======================================================="
echo "Job started at: $(date)"
echo "Job ID: $PBS_JOBID"
echo "Node: $(hostname)"
echo "Available GPUs: $CUDA_VISIBLE_DEVICES"
echo "Working directory: $(pwd)"
echo "Model: $MODEL_NAME"
echo "Data path: $DATA_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "Batch size: $BATCH_SIZE"
echo "Tensor parallel size: $TENSOR_PARALLEL_SIZE"
echo "Checkpoint interval: $CHECKPOINT_INTERVAL"
echo ""

# Check GPU availability
echo "======================================================="
echo "GPU STATUS"
echo "======================================================="
nvidia-smi
echo ""

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Check if data file exists
if [ ! -f "$DATA_PATH" ]; then
    echo "ERROR: Data file $DATA_PATH not found!"
    exit 1
fi

# Check if Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "ERROR: Python script $PYTHON_SCRIPT not found!"
    exit 1
fi

# Main output files
MAIN_OUTPUT="$OUTPUT_DIR/gemma_results_full.pkl"
CHECKPOINT_OUTPUT="$OUTPUT_DIR/gemma_results_full_checkpoint.pkl"

echo "======================================================="
echo "STARTING INFERENCE"
echo "======================================================="

# Check if there's an existing checkpoint and resume accordingly
if [ -f "$CHECKPOINT_OUTPUT" ] || [ -f "$MAIN_OUTPUT" ]; then
    echo "Found existing results. Resuming from checkpoint..."
    RESUME_FLAG="--resume"
else
    echo "No existing results found. Starting fresh..."
    RESUME_FLAG=""
fi

# Build the command
CMD="python3 $PYTHON_SCRIPT \
    --model_name $MODEL_NAME \
    --data_path $DATA_PATH \
    --output_file $MAIN_OUTPUT \
    --batch_size $BATCH_SIZE \
    --tensor_parallel_size $TENSOR_PARALLEL_SIZE \
    --gpu_memory_utilization $GPU_MEMORY_UTIL \
    --checkpoint_interval $CHECKPOINT_INTERVAL"

# Add limit if specified
if [ ! -z "$LIMIT" ]; then
    CMD="$CMD --limit $LIMIT"
fi

# Add resume flag if needed
if [ ! -z "$RESUME_FLAG" ]; then
    CMD="$CMD $RESUME_FLAG"
fi

echo "Executing command:"
echo "$CMD"
echo ""

# Execute the inference
eval $CMD

# Check if the job completed successfully
EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "======================================================="
    echo "INFERENCE COMPLETED SUCCESSFULLY"
    echo "======================================================="
    
    # Generate summary report if results exist
    if [ -f "$MAIN_OUTPUT" ]; then
        echo "Generating summary report..."
        
        # Check results and create summary
        if command -v python3 >/dev/null 2>&1; then
            SUMMARY_FILE="$OUTPUT_DIR/inference_summary.json"
            
            # Create a temporary Python script to generate summary
            cat > /tmp/generate_summary.py << 'EOF'
import pickle
import json
import sys
import os

try:
    output_file = sys.argv[1]
    summary_file = sys.argv[2]
    model_name = sys.argv[3]
    tensor_parallel_size = int(sys.argv[4])
    batch_size = int(sys.argv[5])
    
    with open(output_file, 'rb') as f:
        results = pickle.load(f)
    
    total = len(results)
    success = sum(1 for r in results if r.get('success', False))
    
    summary = {
        'total_samples': total,
        'successful_samples': success,
        'success_rate': round(success/total*100, 2) if total > 0 else 0,
        'model_name': model_name,
        'tensor_parallel_size': tensor_parallel_size,
        'batch_size': batch_size,
        'output_file': output_file,
        'checkpoint_file': output_file.replace('.pkl', '_checkpoint.pkl')
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Summary: {success}/{total} samples processed successfully ({success/total*100:.1f}%)")
    print(f"Summary saved to: {summary_file}")
    
except Exception as e:
    print(f"Error generating summary: {e}")
EOF

            python3 /tmp/generate_summary.py "$MAIN_OUTPUT" "$SUMMARY_FILE" "$MODEL_NAME" "$TENSOR_PARALLEL_SIZE" "$BATCH_SIZE"
            rm -f /tmp/generate_summary.py
        fi
    fi
    
else
    echo ""
    echo "======================================================="
    echo "INFERENCE FAILED"
    echo "======================================================="
    echo "Exit code: $EXIT_CODE"
    echo "Check the log file for error details"
fi

echo ""
echo "======================================================="
echo "JOB SUMMARY"
echo "======================================================="
echo "Job completed at: $(date)"
echo "Job ID: $PBS_JOBID"
echo "Exit code: $EXIT_CODE"
echo "Output directory: $OUTPUT_DIR"
echo "Main results file: $MAIN_OUTPUT"
echo "Checkpoint file: $CHECKPOINT_OUTPUT"

# List output files
if [ -d "$OUTPUT_DIR" ]; then
    echo ""
    echo "Output files:"
    ls -la "$OUTPUT_DIR"
fi

echo "======================================================="

exit $EXIT_CODE