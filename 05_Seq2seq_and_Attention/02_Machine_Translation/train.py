import os
import sys
import time
import math
import json
import random
from collections import Counter
import torch
import torch.nn as nn

from dataset import (
    get_translation_data, get_translation_loaders,
    PAD_ID, SOS_ID, EOS_ID, UNK_ID
)
from models import (
    TranslationEncoder, TranslationStandardDecoder,
    TranslationPeekyDecoder, TranslationSeq2Seq
)


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_corpus_bleu(refs, hyps, max_n=4):
    """
    refs: list of list of word strings
    hyps: list of list of word strings
    """
    if len(refs) == 0 or len(hyps) == 0:
        return 0.0
        
    p_n = [0.0] * max_n
    ref_len_total = 0
    hyp_len_total = 0
    
    for ref, hyp in zip(refs, hyps):
        ref_len_total += len(ref)
        hyp_len_total += len(hyp)
        for n in range(1, max_n + 1):
            ref_ngrams = Counter([tuple(ref[i:i+n]) for i in range(len(ref) - n + 1)])
            hyp_ngrams = Counter([tuple(hyp[i:i+n]) for i in range(len(hyp) - n + 1)])
            
            clipped = sum(min(count, ref_ngrams.get(ng, 0)) for ng, count in hyp_ngrams.items())
            total = max(len(hyp) - n + 1, 0)
            
            # Smoothing
            p_n[n-1] += (clipped + 0.1) / (total + 0.1) if total > 0 else 1.0
            
    p_n = [score / len(refs) for score in p_n]
    
    if hyp_len_total == 0:
        return 0.0
    if hyp_len_total < ref_len_total:
        bp = math.exp(1.0 - ref_len_total / hyp_len_total)
    else:
        bp = 1.0
        
    log_sum = sum((1.0 / max_n) * math.log(max(p, 1e-9)) for p in p_n)
    return bp * math.exp(log_sum) * 100.0


def evaluate_translation_model(model, data_loader, tgt_vocab, device='cpu'):
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    
    total_loss = 0.0
    total_tokens = 0
    
    all_refs = []
    all_hyps = []
    
    short_refs = []
    short_hyps = []
    long_refs = []
    long_hyps = []
    
    exact_matches = 0
    sample_preds = []
    
    with torch.no_grad():
        for p_src, p_dec_in, p_dec_tgt, raw_src, raw_tgt in data_loader:
            p_src = p_src.to(device)
            p_dec_in = p_dec_in.to(device)
            p_dec_tgt = p_dec_tgt.to(device)
            
            # 1. Loss
            logits = model(p_src, p_dec_in)  # (B, T, V)
            loss = criterion(logits.reshape(-1, logits.shape[-1]), p_dec_tgt.reshape(-1))
            
            non_pad = (p_dec_tgt != PAD_ID).sum().item()
            total_loss += loss.item() * non_pad
            total_tokens += non_pad
            
            # 2. Autoregressive greedy generation
            gen_ids = model.generate(p_src, max_len=12, sos_id=SOS_ID, eos_id=EOS_ID)  # (B, max_len)
            
            gen_cpu = gen_ids.cpu().tolist()
            
            for i in range(len(raw_tgt)):
                ref_tokens = raw_tgt[i]
                pred_tokens = tgt_vocab.decode(gen_cpu[i])
                
                all_refs.append(ref_tokens)
                all_hyps.append(pred_tokens)
                
                is_exact = (pred_tokens == ref_tokens)
                if is_exact:
                    exact_matches += 1
                    
                # Short (<= 5 tokens) vs Long (>= 6 tokens)
                if len(ref_tokens) <= 5:
                    short_refs.append(ref_tokens)
                    short_hyps.append(pred_tokens)
                else:
                    long_refs.append(ref_tokens)
                    long_hyps.append(pred_tokens)
                    
                if len(sample_preds) < 10:
                    sample_preds.append({
                        'src': ' '.join(raw_src[i]),
                        'ref': ' '.join(ref_tokens),
                        'hyp': ' '.join(pred_tokens),
                        'exact': is_exact
                    })
                    
    val_loss = total_loss / max(total_tokens, 1)
    corpus_bleu = compute_corpus_bleu(all_refs, all_hyps)
    short_bleu = compute_corpus_bleu(short_refs, short_hyps)
    long_bleu = compute_corpus_bleu(long_refs, long_hyps)
    em_acc = (exact_matches / len(all_refs)) * 100.0 if all_refs else 0.0
    
    return val_loss, corpus_bleu, short_bleu, long_bleu, em_acc, sample_preds


def train_single_translation_model(name, reverse, is_peeky, train_pairs, test_pairs, src_vocab, tgt_vocab,
                                   epochs=20, batch_size=128, lr=0.003, device='cpu'):
    print(f"\n==================================================", flush=True)
    print(f"Starting Translation Model: {name} (Reverse={reverse}, Peeky={is_peeky})", flush=True)
    print(f"==================================================", flush=True)
    
    set_seed(42)
    train_loader, test_loader = get_translation_loaders(
        train_pairs, test_pairs, src_vocab, tgt_vocab, batch_size=batch_size, reverse=reverse
    )
    
    embed_size = 64
    hidden_size = 128
    
    enc = TranslationEncoder(len(src_vocab), embed_size, hidden_size, pad_id=PAD_ID)
    if is_peeky:
        dec = TranslationPeekyDecoder(len(tgt_vocab), embed_size, hidden_size, pad_id=PAD_ID)
    else:
        dec = TranslationStandardDecoder(len(tgt_vocab), embed_size, hidden_size, pad_id=PAD_ID)
        
    model = TranslationSeq2Seq(enc, dec, is_peeky=is_peeky).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model Parameters: {param_count:,}", flush=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    
    history = {
        'name': name,
        'reverse': reverse,
        'is_peeky': is_peeky,
        'param_count': param_count,
        'train_losses': [],
        'val_losses': [],
        'corpus_bleus': [],
        'short_bleus': [],
        'long_bleus': [],
        'em_accs': [],
        'epoch_times': [],
        'sample_preds': []
    }
    
    t_start = time.time()
    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        total_train_loss = 0.0
        total_tokens = 0
        
        for p_src, p_dec_in, p_dec_tgt, _, _ in train_loader:
            p_src, p_dec_in, p_dec_tgt = p_src.to(device), p_dec_in.to(device), p_dec_tgt.to(device)
            optimizer.zero_grad()
            
            logits = model(p_src, p_dec_in)
            loss = criterion(logits.reshape(-1, logits.shape[-1]), p_dec_tgt.reshape(-1))
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            
            non_pad = (p_dec_tgt != PAD_ID).sum().item()
            total_train_loss += loss.item() * non_pad
            total_tokens += non_pad
            
        epoch_time = time.time() - t0
        train_loss = total_train_loss / max(total_tokens, 1)
        
        val_loss, bleu, s_bleu, l_bleu, em, sample_preds = evaluate_translation_model(
            model, test_loader, tgt_vocab, device=device
        )
        
        history['train_losses'].append(train_loss)
        history['val_losses'].append(val_loss)
        history['corpus_bleus'].append(bleu)
        history['short_bleus'].append(s_bleu)
        history['long_bleus'].append(l_bleu)
        history['em_accs'].append(em)
        history['epoch_times'].append(epoch_time)
        history['sample_preds'] = sample_preds
        
        print(f"Epoch {ep:2d}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"BLEU: {bleu:5.2f} (Short: {s_bleu:5.2f}, Long: {l_bleu:5.2f}) | EM: {em:5.2f}% | Time: {epoch_time:.2f}s", flush=True)
              
    history['total_training_time'] = time.time() - t_start
    print(f"Finished {name} in {history['total_training_time']:.1f}s. Final BLEU: {history['corpus_bleus'][-1]:.2f}\n", flush=True)
    return history


def run_translation_experiments(num_samples=15000, epochs=20, batch_size=128, lr=0.003):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}", flush=True)
    
    print("Loading and preparing English-French translation dataset...", flush=True)
    train_pairs, test_pairs, src_vocab, tgt_vocab = get_translation_data(
        num_samples=num_samples, train_ratio=0.9, seed=42, min_len=2, max_len=9
    )
    print(f"Dataset split: {len(train_pairs)} train, {len(test_pairs)} test.", flush=True)
    print(f"Source (Eng) Vocab: {len(src_vocab)}, Target (Fra) Vocab: {len(tgt_vocab)}", flush=True)
    
    experiments = [
        {"name": "Baseline", "reverse": False, "is_peeky": False},
        {"name": "Reverse", "reverse": True, "is_peeky": False},
        {"name": "Peeky", "reverse": False, "is_peeky": True},
        {"name": "Reverse+Peeky", "reverse": True, "is_peeky": True},
    ]
    
    results = {}
    for exp in experiments:
        hist = train_single_translation_model(
            name=exp["name"],
            reverse=exp["reverse"],
            is_peeky=exp["is_peeky"],
            train_pairs=train_pairs,
            test_pairs=test_pairs,
            src_vocab=src_vocab,
            tgt_vocab=tgt_vocab,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            device=device
        )
        results[exp["name"]] = hist
        
    with open('translation_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    print("All translation experiments completed! Results saved to translation_results.json", flush=True)
    return results

if __name__ == '__main__':
    run_translation_experiments(num_samples=15000, epochs=20, batch_size=128, lr=0.003)
