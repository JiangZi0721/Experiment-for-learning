import os
os.environ['MPLCONFIGDIR'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.cache')
import pathlib
if hasattr(pathlib, '_NormalAccessor'):
    pathlib._NormalAccessor.mkdir = staticmethod(os.mkdir)

import json
import numpy as np
import matplotlib.pyplot as plt

def load_results():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    attn_json = os.path.join(cur_dir, 'attention_results.json')
    with open(attn_json, 'r', encoding='utf-8') as f:
        attn_res = json.load(f)
        
    # Also load LSTM baseline and peeky for side-by-side benchmark comparison
    lstm_json = os.path.join(cur_dir, '..', '01_Math_Addition', 'experiment_results.json')
    lstm_res = {}
    if os.path.exists(lstm_json):
        with open(lstm_json, 'r', encoding='utf-8') as f:
            lstm_res = json.load(f)
            
    return attn_res, lstm_res

def plot_learning_curves(attn_res, lstm_res, save_path='attention_comparison.png'):
    epochs = range(1, 26)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=150)
    
    # 1. Exact Match Accuracy
    ax1 = axes[0]
    if 'Baseline' in lstm_res:
        ax1.plot(epochs, [a*100 for a in lstm_res['Baseline']['val_exact_accs']], label='Classic Seq2Seq (Baseline)', color='#7f7f7f', lw=2.0, ls='--')
    if 'Reverse' in lstm_res:
        ax1.plot(epochs, [a*100 for a in lstm_res['Reverse']['val_exact_accs']], label='Reverse Seq2Seq', color='#1f77b4', lw=2.0, ls='--')
    if 'Peeky' in lstm_res:
        ax1.plot(epochs, [a*100 for a in lstm_res['Peeky']['val_exact_accs']], label='Peeky Seq2Seq', color='#ff7f0e', lw=2.2)
    if 'Attention_Normal' in attn_res:
        ax1.plot(epochs, [a*100 for a in attn_res['Attention_Normal']['val_exact_accs']], label='Attention (Normal Input)', color='#d62728', lw=2.5, marker='o')
    if 'Attention_Reverse' in attn_res:
        ax1.plot(epochs, [a*100 for a in attn_res['Attention_Reverse']['val_exact_accs']], label='Attention (Reversed Input)', color='#2ca02c', lw=2.5, marker='s')
        
    ax1.set_title("Exact Match (EM) Accuracy: Attention vs. Peeky vs. Baseline", fontsize=13, fontweight='bold')
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Exact Match Accuracy (%)", fontsize=11)
    ax1.set_ylim(-2, 102)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc='lower right', fontsize=9.5, framealpha=0.9)
    
    # 2. Validation Loss
    ax2 = axes[1]
    if 'Baseline' in lstm_res:
        ax2.plot(epochs, lstm_res['Baseline']['val_losses'], label='Classic Baseline', color='#7f7f7f', lw=2.0, ls='--')
    if 'Peeky' in lstm_res:
        ax2.plot(epochs, lstm_res['Peeky']['val_losses'], label='Peeky', color='#ff7f0e', lw=2.2)
    if 'Attention_Normal' in attn_res:
        ax2.plot(epochs, attn_res['Attention_Normal']['val_losses'], label='Attention (Normal)', color='#d62728', lw=2.5)
    if 'Attention_Reverse' in attn_res:
        ax2.plot(epochs, attn_res['Attention_Reverse']['val_losses'], label='Attention (Reverse)', color='#2ca02c', lw=2.5)
        
    ax2.set_title("Validation Loss (Cross Entropy) vs Epoch", fontsize=13, fontweight='bold')
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Loss", fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(loc='upper right', fontsize=9.5, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved learning curves to {save_path}")

def plot_attention_heatmaps(attn_res, save_path='attention_heatmaps.png'):
    """
    Saito Ch08: Visualizing Attention Weight Alignment Heatmap
    """
    normal_samples = attn_res['Attention_Normal'].get('sample_preds', [])
    if not normal_samples:
        return
        
    # Select 4 distinct samples
    selected = normal_samples[:4]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=150)
    axes = axes.flatten()
    
    for i, sample in enumerate(selected):
        ax = axes[i]
        attn_matrix = np.array(sample['attention_matrix'])  # (4, 7)
        q_chars = list(sample['question'])                  # 7 input tokens
        pred_chars = list(sample['pred'])                   # 4 output tokens
        
        # Plot heatmap
        im = ax.imshow(attn_matrix, cmap='viridis', aspect='auto', vmin=0.0, vmax=1.0)
        
        ax.set_xticks(range(len(q_chars)))
        ax.set_yticks(range(len(pred_chars)))
        ax.set_xticklabels([repr(c)[1:-1] for c in q_chars], fontsize=12, fontweight='bold')
        ax.set_yticklabels([repr(c)[1:-1] for c in pred_chars], fontsize=12, fontweight='bold')
        
        status = "Correct" if sample['correct'] else "Error"
        ax.set_title(f"Input: {repr(sample['question'])} -> Pred: {repr(sample['pred'])} ({status})", fontsize=11, fontweight='bold')
        ax.set_xlabel("Input Characters (Source)", fontsize=10)
        ax.set_ylabel("Generated Characters (Target)", fontsize=10)
        
        # Overlay values
        for r in range(len(pred_chars)):
            for c in range(len(q_chars)):
                val = attn_matrix[r, c]
                color = "white" if val > 0.4 else "black"
                ax.text(c, r, f"{val:.2f}", ha="center", va="center", color=color, fontsize=8)
                
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7, label='Attention Weight')
    plt.suptitle("Attention Soft-Alignment Matrix Heatmaps (Saito Ch08)", fontsize=14, fontweight='bold')
    plt.savefig(save_path)
    plt.close()
    print(f"Saved attention heatmaps to {save_path}")

def generate_summary(attn_res, lstm_res, out_file='summary_report.md'):
    lines = []
    lines.append("# Attention Seq2Seq Comprehensive Benchmark (Saito Ch08 vs. Peeky vs. Baseline)")
    lines.append("")
    lines.append("| Model | Params | Training Time (s) | Final Loss | Final EM Acc (%) | Carry Acc (%) | No-Carry Acc (%) |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    
    # 1. Baseline & Peeky from Exp 01
    if 'Baseline' in lstm_res:
        d = lstm_res['Baseline']
        lines.append(f"| **Classic Baseline** | {d['param_count']:,} | {d['total_training_time']:.1f} | {d['val_losses'][-1]:.4f} | {d['val_exact_accs'][-1]*100:.2f}% | {d['subgroup_accs'][-1]['carry']*100:.2f}% | {d['subgroup_accs'][-1]['no_carry']*100:.2f}% |")
    if 'Reverse' in lstm_res:
        d = lstm_res['Reverse']
        lines.append(f"| **Reverse Seq2Seq** | {d['param_count']:,} | {d['total_training_time']:.1f} | {d['val_losses'][-1]:.4f} | {d['val_exact_accs'][-1]*100:.2f}% | {d['subgroup_accs'][-1]['carry']*100:.2f}% | {d['subgroup_accs'][-1]['no_carry']*100:.2f}% |")
    if 'Peeky' in lstm_res:
        d = lstm_res['Peeky']
        lines.append(f"| **Peeky Seq2Seq** | {d['param_count']:,} | {d['total_training_time']:.1f} | {d['val_losses'][-1]:.4f} | {d['val_exact_accs'][-1]*100:.2f}% | {d['subgroup_accs'][-1]['carry']*100:.2f}% | {d['subgroup_accs'][-1]['no_carry']*100:.2f}% |")
    if 'Reverse+Peeky' in lstm_res:
        d = lstm_res['Reverse+Peeky']
        lines.append(f"| **Reverse+Peeky** | {d['param_count']:,} | {d['total_training_time']:.1f} | {d['val_losses'][-1]:.4f} | {d['val_exact_accs'][-1]*100:.2f}% | {d['subgroup_accs'][-1]['carry']*100:.2f}% | {d['subgroup_accs'][-1]['no_carry']*100:.2f}% |")
        
    # 2. Attention models
    for name, d in attn_res.items():
        lines.append(f"| **{name}** | {d['param_count']:,} | {d['total_training_time']:.1f} | {d['val_losses'][-1]:.4f} | **{d['val_exact_accs'][-1]*100:.2f}%** | {d['carry_accs'][-1]*100:.2f}% | {d['no_carry_accs'][-1]*100:.2f}% |")
        
    lines.append("")
    lines.append("## Attention Model Sample Predictions")
    lines.append("")
    for name, d in attn_res.items():
        lines.append(f"### Model: {name}")
        lines.append("| Input Question | Ground Truth | Predicted | Correct? | Has Carry? |")
        lines.append("| :--- | :---: | :---: | :---: | :---: |")
        for sample in d.get('sample_preds', [])[:6]:
            status = "✅" if sample['correct'] else "❌"
            lines.append(f"| `{sample['question']}` | `{sample['target']}` | `{sample['pred']}` | {status} | {sample['has_carry']} |")
        lines.append("")
        
    with open(out_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    print(f"Generated summary table at {out_file}")

def main():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    attn_res, lstm_res = load_results()
    plot_learning_curves(attn_res, lstm_res, os.path.join(cur_dir, 'attention_comparison.png'))
    plot_attention_heatmaps(attn_res, os.path.join(cur_dir, 'attention_heatmaps.png'))
    generate_summary(attn_res, lstm_res, os.path.join(cur_dir, 'summary_report.md'))

if __name__ == '__main__':
    main()
