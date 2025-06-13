
import gc
def compute_accuracy(eval_preds, tokenizer, id2label):
    logits, labels = eval_preds
    
    if len(logits.shape) == 3:  # (batch_size, seq_len, vocab_size)
        
        batch_size = logits.shape[0]
        preds = []
        
        for i in range(batch_size):
            
            valid_positions = labels[i] != -100
            if valid_positions.any():
                last_valid_pos = valid_positions.nonzero()[-1].item()
                pred_token = logits[i, last_valid_pos].argmax().item()
                preds.append(pred_token)
            else:
                preds.append(tokenizer.pad_token_id)
        
      
        true_labels = []
        for i in range(batch_size):
            valid_positions = labels[i] != -100
            if valid_positions.any():
                last_valid_pos = valid_positions.nonzero()[-1].item()
                true_labels.append(labels[i, last_valid_pos].item())
            else:
                true_labels.append(tokenizer.pad_token_id)
        
        
        correct = sum(p == l for p, l in zip(preds, true_labels))
        accuracy = correct / len(preds) if preds else 0.0
        
    else:
      
        preds = logits.argmax(axis=-1)
        valid_mask = labels != -100
        if valid_mask.any():
            accuracy = (preds[valid_mask] == labels[valid_mask]).mean()
        else:
            accuracy = 0.0
    
  
    gc.collect()
    
    return {"accuracy": float(accuracy)}