def compute_accuracy(eval_preds, tokenizer, id2label):
    logits, labels = eval_preds
    
    # 立即释放不需要的内存
    import gc
    
    # 只取最后一个时间步的logits进行预测（对于生成任务）
    if len(logits.shape) == 3:  # (batch_size, seq_len, vocab_size)
        # 找到每个序列中最后一个非padding的位置
        batch_size = logits.shape[0]
        preds = []
        
        for i in range(batch_size):
            # 找到当前序列中最后一个有效标签的位置
            valid_positions = labels[i] != -100
            if valid_positions.any():
                last_valid_pos = valid_positions.nonzero()[-1].item()
                pred_token = logits[i, last_valid_pos].argmax().item()
                preds.append(pred_token)
            else:
                preds.append(tokenizer.pad_token_id)
        
        # 提取对应的真实标签
        true_labels = []
        for i in range(batch_size):
            valid_positions = labels[i] != -100
            if valid_positions.any():
                last_valid_pos = valid_positions.nonzero()[-1].item()
                true_labels.append(labels[i, last_valid_pos].item())
            else:
                true_labels.append(tokenizer.pad_token_id)
        
        # 计算准确率
        correct = sum(p == l for p, l in zip(preds, true_labels))
        accuracy = correct / len(preds) if preds else 0.0
        
    else:
        # 如果logits已经是2D的
        preds = logits.argmax(axis=-1)
        valid_mask = labels != -100
        if valid_mask.any():
            accuracy = (preds[valid_mask] == labels[valid_mask]).mean()
        else:
            accuracy = 0.0
    
    # 强制垃圾回收
    gc.collect()
    
    return {"accuracy": float(accuracy)}