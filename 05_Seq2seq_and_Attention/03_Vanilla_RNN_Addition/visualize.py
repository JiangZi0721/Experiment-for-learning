import os
os.environ['MPLCONFIGDIR'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.cache')
import pathlib
if hasattr(pathlib, '_NormalAccessor'):
    pathlib._NormalAccessor.mkdir = staticmethod(os.mkdir)

import json
import numpy as np
import matplotlib.pyplot as plt

def load_results(json_path='rnn_results.json'):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def plot_curves(results, save_path='rnn_accuracy_comparison.png'):
    epochs = range(1, len(next(iter(results.values()))['val_exact_accs']) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=150)
    
    colors = {
        'RNN_Baseline': '#7f7f7f',
        'RNN_Reverse': '#1f77b4',
        'RNN_Peeky': '#ff7f0e',
        'RNN_Reverse+Peeky': '#2ca02c'
    }
    
    # 1. Exact Match Accuracy
    ax1 = axes[0]
    for name, data in results.items():
        accs = [acc * 100 for acc in data['val_exact_accs']]
        ax1.plot(epochs, accs, label=name, color=colors.get(name, 'black'), lw=2.2, marker='o' if len(epochs)<=15 else None)
        
    ax1.set_title("Vanilla RNN: Validation Exact Match (EM) Accuracy vs Epoch", fontsize=13, fontweight='bold')
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Exact Match Accuracy (%)", fontsize=11)
    ax1.set_ylim(-2, 102)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc='lower right', fontsize=10, framealpha=0.9)
    
    # 2. Training Loss
    ax2 = axes[1]
    for name, data in results.items():
        losses = data['train_losses']
        ax2.plot(epochs, losses, label=name, color=colors.get(name, 'black'), lw=2.2)
        
    ax2.set_title("Vanilla RNN: Training Loss vs Epoch", fontsize=13, fontweight='bold')
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Loss", fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved RNN learning curves to {save_path}")

def generate_summary_table(results, out_file='summary_report.md'):
    lines = []
    lines.append("# Vanilla RNN Seq2Seq Optimization Experiment Summary (Arithmetic Addition)")
    lines.append("")
    lines.append("| Model | Params | Training Time (s) | Final Loss | Final EM Acc (%) | Carry Acc (%) | No-Carry Acc (%) |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    
    for name, data in results.items():
        params = f"{data['param_count']:,}"
        time_s = f"{data.get('total_training_time', 0):.1f}"
        final_loss = f"{data['val_losses'][-1]:.4f}"
        final_em = f"{data['val_exact_accs'][-1]*100:.2f}"
        carry_acc = f"{data['carry_accs'][-1]*100:.2f}"
        no_carry_acc = f"{data['no_carry_accs'][-1]*100:.2f}"
        
        lines.append(f"| **{name}** | {params} | {time_s} | {final_loss} | **{final_em}%** | {carry_acc}% | {no_carry_acc}% |")
        
    lines.append("")
    lines.append("## Sample Predictions")
    lines.append("")
    for name, data in results.items():
        lines.append(f"### Model: {name}")
        lines.append("| Input Question | Ground Truth | Predicted | Correct? | Has Carry? |")
        lines.append("| :--- | :---: | :---: | :---: | :---: |")
        for sample in data.get('sample_preds', [])[:6]:
            status = "✅" if sample['correct'] else "❌"
            lines.append(f"| `{sample['question']}` | `{sample['target']}` | `{sample['pred']}` | {status} | {sample['has_carry']} |")
        lines.append("")
        
    summary_text = "\n".join(lines)
    with open(out_file, 'w', encoding='utf-8') as f:
        f.write(summary_text)
    print(f"Generated summary table at {out_file}")

def main():
    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rnn_results.json')
    if not os.path.exists(json_path):
        print("rnn_results.json not found.")
        return
    results = load_results(json_path)
    plot_curves(results, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rnn_accuracy_comparison.png'))
    generate_summary_table(results, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'summary_report.md'))

if __name__ == '__main__':
    main()
