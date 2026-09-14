# Attention Seq2Seq Comprehensive Benchmark (Saito Ch08 vs. Peeky vs. Baseline)

| Model | Params | Training Time (s) | Final Loss | Final EM Acc (%) | Carry Acc (%) | No-Carry Acc (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Classic Baseline** | 151,597 | 76.5 | 0.6000 | 26.26% | 27.25% | 21.28% |
| **Reverse Seq2Seq** | 151,597 | 60.1 | 0.5614 | 23.22% | 23.01% | 24.30% |
| **Peeky Seq2Seq** | 218,797 | 63.9 | 0.0320 | 97.06% | 97.51% | 94.80% |
| **Reverse+Peeky** | 218,797 | 94.2 | 0.2409 | 66.80% | 68.03% | 60.58% |
| **Attention_Normal** | 153,261 | 81.2 | 0.4489 | **37.92%** | 37.43% | 40.39% |
| **Attention_Reverse** | 153,261 | 74.0 | 0.0134 | **99.22%** | 99.47% | 97.94% |

## Attention Model Sample Predictions

### Model: Attention_Normal
| Input Question | Ground Truth | Predicted | Correct? | Has Carry? |
| :--- | :---: | :---: | :---: | :---: |
| `426+11 ` | `437 ` | `445 ` | ❌ | False |
| `211+903` | `1114` | `1114` | ✅ | True |
| `840+444` | `1284` | `1274` | ❌ | True |
| `581+111` | `692 ` | `691 ` | ❌ | False |
| `441+311` | `752 ` | `741 ` | ❌ | False |
| `473+589` | `1062` | `1062` | ✅ | True |

### Model: Attention_Reverse
| Input Question | Ground Truth | Predicted | Correct? | Has Carry? |
| :--- | :---: | :---: | :---: | :---: |
| `426+11 ` | `437 ` | `437 ` | ✅ | False |
| `211+903` | `1114` | `1114` | ✅ | True |
| `840+444` | `1284` | `1284` | ✅ | True |
| `581+111` | `692 ` | `692 ` | ✅ | False |
| `441+311` | `752 ` | `752 ` | ✅ | False |
| `473+589` | `1062` | `1062` | ✅ | True |
