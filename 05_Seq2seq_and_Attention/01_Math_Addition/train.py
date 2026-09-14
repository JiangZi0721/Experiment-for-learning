import os
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn
from dataset import get_dataloaders, CHAR_TO_ID, ID_TO_CHAR
from models import Encoder, StandardDecoder, PeekyDecoder, Seq2Seq

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def has_carry(q_str):
    """
    Checks if an addition A+B has any carry.
    q_str: '16+75  '
    """
    clean_q = q_str.strip()
    if '+' not in clean_q:
        return False
    parts = clean_q.split('+')
    a_str, b_str = parts[0], parts[1]
    
    # Check carry digit by digit from right to left
    a_rev = [int(c) for c in reversed(a_str)]
    b_rev = [int(c) for c in reversed(b_str)]
    max_len = max(len(a_rev), len(b_rev))
    
    carry = 0
    for i in range(max_len):
        d1 = a_rev[i] if i < len(a_rev) else 0
        d2 = b_rev[i] if i < len(b_rev) else 0
        s = d1 + d2 + carry
        if s >= 10:
            return True
        carry = s // 10
    return False

def get_max_digits(q_str):
    """Returns max digit length of operands."""
    clean_q = q_str.strip()
    parts = clean_q.split('+')
    return max(len(parts[0]), len(parts[1]))

def evaluate_model(model, data_loader, raw_questions, raw_answers, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    total_loss = 0.0
    total_samples = 0
    correct_exact = 0
    correct_tokens = 0
    total_tokens = 0
    
    # Subgroup tracking
    subgroups = {
        'no_carry': {'total': 0, 'correct': 0},
        'carry': {'total': 0, 'correct': 0},
        '1_digit': {'total': 0, 'correct': 0},
        '2_digit': {'total': 0, 'correct': 0},
        '3_digit': {'total': 0, 'correct': 0},
    }
    
    sample_preds = []
    
    idx_offset = 0
    with torch.no_grad():
        for bx, by in data_loader:
            bx = bx.to(device)
            by = by.to(device)
            batch_size = bx.shape[0]
            
            # 1. Compute loss
            logits = model(bx, by)  # (B, 4, V)
            loss = criterion(logits.reshape(-1, logits.shape[-1]), by[:, 1:].reshape(-1))
            total_loss += loss.item() * batch_size
            
            # 2. Autoregressive greedy generation
            gen_ids = model.generate(bx, max_len=4, start_id=CHAR_TO_ID['_'])  # (B, 4)
            
            # Token accuracy
            target_tokens = by[:, 1:]
            correct_tokens += (gen_ids == target_tokens).sum().item()
            total_tokens += target_tokens.numel()
            
            # Exact match string comparison
            gen_cpu = gen_ids.cpu().numpy()
            tgt_cpu = target_tokens.cpu().numpy()
            
            for i in range(batch_size):
                sample_idx = idx_offset + i
                raw_q = raw_questions[sample_idx]
                raw_a = raw_answers[sample_idx]  # e.g. '_91  '
                
                gen_str = ''.join([ID_TO_CHAR[cid] for cid in gen_cpu[i]])
                true_str = raw_a[1:]  # e.g. '91  '
                
                is_correct = (gen_str == true_str)
                if is_correct:
                    correct_exact += 1
                
                # Subgroup: carry vs no carry
                is_c = has_carry(raw_q)
                grp_carry = 'carry' if is_c else 'no_carry'
                subgroups[grp_carry]['total'] += 1
                if is_correct:
                    subgroups[grp_carry]['correct'] += 1
                    
                # Subgroup: digit length
                max_d = get_max_digits(raw_q)
                grp_d = f'{max_d}_digit' if f'{max_d}_digit' in subgroups else '3_digit'
                subgroups[grp_d]['total'] += 1
                if is_correct:
                    subgroups[grp_d]['correct'] += 1
                    
                # Save a few sample predictions
                if len(sample_preds) < 10:
                    sample_preds.append({
                        'question': raw_q,
                        'target': true_str,
                        'pred': gen_str,
                        'correct': is_correct,
                        'has_carry': is_c,
                        'max_digits': max_d
                    })
                    
            total_samples += batch_size
            idx_offset += batch_size
            
    val_loss = total_loss / total_samples
    exact_acc = correct_exact / total_samples
    token_acc = correct_tokens / total_tokens
    
    subgroup_acc = {}
    for k, v in subgroups.items():
        subgroup_acc[k] = (v['correct'] / v['total']) if v['total'] > 0 else 0.0
        
    return val_loss, exact_acc, token_acc, subgroup_acc, sample_preds


def train_single_model(name, reverse, is_peeky, num_samples=50000, epochs=25, batch_size=256, lr=0.002, device='cpu'):
    print(f"\n==================================================")
    print(f"Starting Training: {name} (Reverse={reverse}, Peeky={is_peeky})")
    print(f"==================================================")
    
    # 1. Dataset
    set_seed(42)
    train_loader, test_loader, (train_q, train_a), (test_q, test_a) = get_dataloaders(
        num_samples=num_samples, train_ratio=0.9, batch_size=batch_size, reverse=reverse, seed=42
    )
    
    # 2. Model
    set_seed(42)
    vocab_size = len(CHAR_TO_ID)
    embed_size = 16
    hidden_size = 128
    
    enc = Encoder(vocab_size, embed_size, hidden_size)
    if is_peeky:
        dec = PeekyDecoder(vocab_size, embed_size, hidden_size)
    else:
        dec = StandardDecoder(vocab_size, embed_size, hidden_size)
        
    model = Seq2Seq(enc, dec, is_peeky=is_peeky).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model Parameters: {param_count:,}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    history = {
        'name': name,
        'reverse': reverse,
        'is_peeky': is_peeky,
        'param_count': param_count,
        'train_losses': [],
        'val_losses': [],
        'val_exact_accs': [],
        'val_token_accs': [],
        'subgroup_accs': [],
        'epoch_times': [],
        'sample_preds': []
    }
    
    total_start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        total_train_loss = 0.0
        total_train_samples = 0
        
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            
            logits = model(bx, by)  # (B, 4, V)
            loss = criterion(logits.reshape(-1, vocab_size), by[:, 1:].reshape(-1))
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            
            total_train_loss += loss.item() * bx.shape[0]
            total_train_samples += bx.shape[0]
            
        epoch_time = time.time() - t0
        train_loss = total_train_loss / total_train_samples
        
        # Evaluate on test set
        val_loss, exact_acc, token_acc, subgroup_acc, sample_preds = evaluate_model(
            model, test_loader, test_q, test_a, device
        )
        
        history['train_losses'].append(train_loss)
        history['val_losses'].append(val_loss)
        history['val_exact_accs'].append(exact_acc)
        history['val_token_accs'].append(token_acc)
        history['subgroup_accs'].append(subgroup_acc)
        history['epoch_times'].append(epoch_time)
        history['sample_preds'] = sample_preds
        
        print(f"Epoch {epoch:2d}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Exact Match: {exact_acc*100:5.2f}% | Token Acc: {token_acc*100:5.2f}% | "
              f"Carry: {subgroup_acc['carry']*100:5.2f}% | No-Carry: {subgroup_acc['no_carry']*100:5.2f}% | Time: {epoch_time:.2f}s", flush=True)
              
    history['total_training_time'] = time.time() - total_start_time
    print(f"Finished {name} in {history['total_training_time']:.1f}s. Final Exact Match: {history['val_exact_accs'][-1]*100:.2f}%\n", flush=True)

    
    return history


def run_all_experiments(epochs=25, num_samples=50000, batch_size=256, lr=0.002):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}", flush=True)
    
    experiments = [
        {"name": "Baseline", "reverse": False, "is_peeky": False},
        {"name": "Reverse", "reverse": True, "is_peeky": False},
        {"name": "Peeky", "reverse": False, "is_peeky": True},
        {"name": "Reverse+Peeky", "reverse": True, "is_peeky": True},
    ]
    
    results = {}
    for exp in experiments:
        hist = train_single_model(
            name=exp["name"],
            reverse=exp["reverse"],
            is_peeky=exp["is_peeky"],
            num_samples=num_samples,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            device=device
        )
        results[exp["name"]] = hist
        
    # Save results to json
    with open('experiment_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    print("All experiments completed! Results saved to experiment_results.json", flush=True)
    return results

if __name__ == '__main__':
    run_all_experiments(epochs=25, num_samples=50000, batch_size=256, lr=0.002)

