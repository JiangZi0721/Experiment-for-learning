import os
os.environ['MPLCONFIGDIR'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.cache')
import pathlib
if hasattr(pathlib, '_NormalAccessor'):
    pathlib._NormalAccessor.mkdir = staticmethod(os.mkdir)

import json
import numpy as np
import matplotlib.pyplot as plt

def load_translation_results(json_path='translation_results.json'):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def plot_translation_curves(results, save_path='translation_bleu_comparison.png'):
    epochs = range(1, len(next(iter(results.values()))['corpus_bleus']) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=150)
    
    colors = {
        'Baseline': '#7f7f7f',
        'Reverse': '#1f77b4',
        'Peeky': '#ff7f0e',
        'Reverse+Peeky': '#2ca02c'
    }
    
    # 1. BLEU Score
    ax1 = axes[0]
    for name, data in results.items():
        bleus = data['corpus_bleus']
        ax1.plot(epochs, bleus, label=name, color=colors.get(name, 'black'), lw=2.2, marker='o' if len(epochs)<=20 else None)
        
    ax1.set_title("Validation BLEU Score vs Epoch (English -> French)", fontsize=13, fontweight='bold')
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Corpus BLEU Score", fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc='lower right', fontsize=10, framealpha=0.9)
    
    # 2. Validation Loss
    ax2 = axes[1]
    for name, data in results.items():
        losses = data['val_losses']
        ax2.plot(epochs, losses, label=name, color=colors.get(name, 'black'), lw=2.2)
        
    ax2.set_title("Validation Loss (Cross Entropy) vs Epoch", fontsize=13, fontweight='bold')
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Validation Loss", fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved translation learning curves to {save_path}")

def plot_translation_length_analysis(results, save_path='translation_length_analysis.png'):
    model_names = list(results.keys())
    
    overall_bleus = [results[m]['corpus_bleus'][-1] for m in model_names]
    short_bleus = [results[m]['short_bleus'][-1] for m in model_names]
    long_bleus = [results[m]['long_bleus'][-1] for m in model_names]
    
    x = np.arange(len(model_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    
    rects1 = ax.bar(x - width, overall_bleus, width, label='Overall BLEU', color='#4c72b0')
    rects2 = ax.bar(x, short_bleus, width, label='Short (<= 5 tokens)', color='#55a868')
    rects3 = ax.bar(x + width, long_bleus, width, label='Long (>= 6 tokens)', color='#c44e52')
    
    ax.set_title("Translation BLEU Score by Sentence Length", fontsize=13, fontweight='bold')
    ax.set_ylabel("BLEU Score", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=11, fontweight='bold')
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    ax.legend(loc='upper left', fontsize=10)
    
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved translation length analysis to {save_path}")

def generate_translation_summary(results, out_file='summary_report.md'):

    lines = []
    lines.append("# Machine Translation Experiment Summary (English -> French)")
    lines.append("")
    lines.append("| Model | Params | Training Time (s) | Final Val Loss | Final BLEU | Short BLEU (<=5) | Long BLEU (>=6) | Exact Match (%) |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    
    for name, data in results.items():
        params = f"{data['param_count']:,}"
        time_s = f"{data.get('total_training_time', 0):.1f}"
        final_loss = f"{data['val_losses'][-1]:.4f}"
        final_bleu = f"{data['corpus_bleus'][-1]:.2f}"
        short_bleu = f"{data['short_bleus'][-1]:.2f}"
        long_bleu = f"{data['long_bleus'][-1]:.2f}"
        em = f"{data['em_accs'][-1]:.2f}%"
        
        lines.append(f"| **{name}** | {params} | {time_s} | {final_loss} | **{final_bleu}** | {short_bleu} | {long_bleu} | {em} |")
        
    lines.append("")
    lines.append("## Sample Translation Predictions")
    lines.append("")
    for name, data in results.items():
        lines.append(f"### Model: {name}")
        lines.append("| Source (English) | Reference (French) | Prediction (Hypothesis) | Exact Match? |")
        lines.append("| :--- | :--- | :--- | :---: |")
        for sample in data.get('sample_preds', [])[:6]:
            status = "✅" if sample['exact'] else "❌"
            lines.append(f"| `{sample['src']}` | `{sample['ref']}` | `{sample['hyp']}` | {status} |")
        lines.append("")
        
    summary_text = "\n".join(lines)
    with open(out_file, 'w', encoding='utf-8') as f:
        f.write(summary_text)
    print(f"Generated summary table at {out_file}")

def main():
    if not os.path.exists('translation_results.json'):
        print("translation_results.json not found.")
        return
    results = load_translation_results('translation_results.json')
    plot_translation_curves(results)
    plot_translation_length_analysis(results)
    generate_translation_summary(results)

if __name__ == '__main__':
    main()
