import os
os.environ['MPLCONFIGDIR'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.cache')
import pathlib
if hasattr(pathlib, '_NormalAccessor'):
    pathlib._NormalAccessor.mkdir = staticmethod(os.mkdir)

import json
import numpy as np
import matplotlib.pyplot as plt

def load_results(json_path='experiment_results.json'):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def plot_curves(results, save_path='accuracy_comparison.png'):
    epochs = range(1, len(next(iter(results.values()))['val_exact_accs']) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=150)
    
    colors = {
        'Baseline': '#7f7f7f',
        'Reverse': '#1f77b4',
        'Peeky': '#ff7f0e',
        'Reverse+Peeky': '#2ca02c'
    }
    
    # 1. Exact Match Accuracy
    ax1 = axes[0]
    for name, data in results.items():
        accs = [acc * 100 for acc in data['val_exact_accs']]
        ax1.plot(epochs, accs, label=name, color=colors.get(name, 'black'), lw=2.2, marker='o' if len(epochs)<=15 else None)
        
    ax1.set_title("Validation Exact Match (EM) Accuracy vs Epoch", fontsize=13, fontweight='bold')
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
        
    ax2.set_title("Training Loss (Cross Entropy) vs Epoch", fontsize=13, fontweight='bold')
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Loss", fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved learning curves to {save_path}")

def plot_subgroup_analysis(results, save_path='subgroup_carry_analysis.png'):
    model_names = list(results.keys())
    
    overall_accs = [results[m]['val_exact_accs'][-1] * 100 for m in model_names]
    carry_accs = [results[m]['subgroup_accs'][-1]['carry'] * 100 for m in model_names]
    no_carry_accs = [results[m]['subgroup_accs'][-1]['no_carry'] * 100 for m in model_names]
    
    x = np.arange(len(model_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    
    rects1 = ax.bar(x - width, overall_accs, width, label='Overall (All)', color='#4c72b0')
    rects2 = ax.bar(x, carry_accs, width, label='With Carry', color='#c44e52')
    rects3 = ax.bar(x + width, no_carry_accs, width, label='No Carry', color='#55a868')
    
    ax.set_title("Final Exact Match Accuracy Breakdown: Carry vs. No-Carry", fontsize=13, fontweight='bold')

    ax.set_ylabel("Exact Match Accuracy (%)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=11, fontweight='bold')
    ax.set_ylim(0, 110)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    
    # Attach labels above bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, rotation=0)

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved carry subgroup analysis to {save_path}")

def generate_summary_table(results, out_file='summary_report.md'):

    lines = []
    lines.append("# Seq2Seq Optimization Experiment Summary (LSTM on Arithmetic Addition)")
    lines.append("")
    lines.append("| Model | Params | Training Time (s) | Final Loss | Final EM Acc (%) | Carry Acc (%) | No-Carry Acc (%) | Reached 50% Epoch | Reached 90% Epoch |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    
    for name, data in results.items():
        params = f"{data['param_count']:,}"
        time_s = f"{data.get('total_training_time', 0):.1f}"
        final_loss = f"{data['val_losses'][-1]:.4f}"
        final_em = f"{data['val_exact_accs'][-1]*100:.2f}"
        carry_acc = f"{data['subgroup_accs'][-1]['carry']*100:.2f}"
        no_carry_acc = f"{data['subgroup_accs'][-1]['no_carry']*100:.2f}"
        
        # Epochs to reach milestones
        ep50 = '-'
        ep90 = '-'
        for ep, acc in enumerate(data['val_exact_accs'], start=1):
            if ep50 == '-' and acc >= 0.50:
                ep50 = str(ep)
            if ep90 == '-' and acc >= 0.90:
                ep90 = str(ep)
                
        lines.append(f"| **{name}** | {params} | {time_s} | {final_loss} | **{final_em}%** | {carry_acc}% | {no_carry_acc}% | {ep50} | {ep90} |")
        
    lines.append("")
    lines.append("## Sample Predictions & Qualitative Error Analysis")
    lines.append("")
    for name, data in results.items():
        lines.append(f"### Model: {name}")
        lines.append("| Input Question | Ground Truth | Predicted | Correct? | Has Carry? | Digits |")
        lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
        for sample in data.get('sample_preds', [])[:6]:
            status = "✅" if sample['correct'] else "❌"
            lines.append(f"| `{sample['question']}` | `{sample['target']}` | `{sample['pred']}` | {status} | {sample['has_carry']} | {sample['max_digits']} |")
        lines.append("")
        
    summary_text = "\n".join(lines)
    with open(out_file, 'w', encoding='utf-8') as f:
        f.write(summary_text)
    print(f"Generated summary table at {out_file}")

def main():
    if not os.path.exists('experiment_results.json'):
        print("experiment_results.json not found yet.")
        return
    results = load_results('experiment_results.json')
    plot_curves(results)
    plot_subgroup_analysis(results)
    generate_summary_table(results)

if __name__ == '__main__':
    main()
