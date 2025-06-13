# how to run

download dataset into current project dir: ./data/train_samples.pkl and ./data/val_samples.pkl


## run python ./train_router_selector.py
go to parse_args(): change pretrain model version


## run python evaluate_model.py to inference 
here change defalut hyperparameter
name = "router_selector_4b"
base_model_name
base_model_name = 'google/gemma-3-4b-it'
lora_checkpoint_path = f'./outputs/{name}/checkpoint-29242'