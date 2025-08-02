
# Full Evaluation Job (PBS Script)

This job script runs full evaluations of a fine-tuned **Gemma 3 4B** model on two benchmarks: **HELMET** and **InfiniteBench**, using a multi-GPU setup.

---

## 🛠️ Job Configuration (PBS)

- **Output Log**: `full_eval_{dataset_name}.out`
- **Error Log**: `full_eval_{dataset_name}.err`
- **Runtime Limit**: 24 hours
- **Node**: `gpu008`
- **Queue**: `poderoso`
- **GPUs Used**: GPU 1 and 2

---

## 📦 Data Preparation

### HELMET
```bash
git clone https://github.com/princeton-nlp/HELMET.git
cd HELMET
bash scripts/download_data.sh
```

### InfiniteBench
```bash
git clone https://github.com/OpenBMB/InfiniteBench.git
cd InfiniteBench
bash scripts/download_dataset.sh
```

---

## 🚀 Inference on HELMET

### Option 1: Use PBS script
```bash
qsub run_infer_helmet.sh
```

### Option 2: Run directly
```bash
python inference_helmet.py \
    --checkpoint_path ./experiments/google_gemma-3-4b-it_20250730_221314/final_model/ \
    --data_dir ../HELMET/data/ \
    --output_prefix results_HELMET \
    --save_raw_text \
    --batch_size 200 \
    --tensor_parallel_size 2 \
    --save_interval 500
```

---

## 🚀 Inference on InfiniteBench

### Option 1: Use PBS script
```bash
qsub run_infer_infinitebench.sh
```

### Option 2: Run directly
```bash
python inference_infinitebench.py \
    --checkpoint_path ./experiments/google_gemma-3-4b-it_20250730_221314/final_model/ \
    --data_dir ../InfiniteBench/data/ \
    --output_prefix results_infinitebench \
    --save_raw_text \
    --batch_size 400 \
    --tensor_parallel_size 4 \
    --save_interval 500
    # --num_samples 1  # Optional: limit to 1 sample per input
```

---

## 📌 Notes

- Adjust `--batch_size` and `--tensor_parallel_size` according to available GPU memory.
- Use `--save_raw_text` to preserve raw model outputs for later evaluation or debugging.
