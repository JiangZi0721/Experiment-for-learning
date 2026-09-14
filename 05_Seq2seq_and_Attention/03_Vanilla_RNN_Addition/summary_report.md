# Vanilla RNN Seq2Seq Optimization Experiment Summary (Arithmetic Addition)

| Model | Params | Training Time (s) | Final Loss | Final EM Acc (%) | Carry Acc (%) | No-Carry Acc (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **RNN_Baseline** | 39,469 | 48.2 | 0.4623 | **30.18%** | 31.70% | 22.49% |
| **RNN_Reverse** | 39,469 | 46.6 | 0.0702 | **92.72%** | 93.79% | 87.30% |
| **RNN_Peeky** | 57,517 | 36.9 | 0.0928 | **87.82%** | 88.86% | 82.59% |
| **RNN_Reverse+Peeky** | 57,517 | 39.4 | 0.0485 | **94.10%** | 95.06% | 89.24% |

## Sample Predictions

### Model: RNN_Baseline
| Input Question | Ground Truth | Predicted | Correct? | Has Carry? |
| :--- | :---: | :---: | :---: | :---: |
| `426+11 ` | `437 ` | `441 ` | ❌ | False |
| `211+903` | `1114` | `1116` | ❌ | True |
| `840+444` | `1284` | `1283` | ❌ | True |
| `581+111` | `692 ` | `692 ` | ✅ | False |
| `441+311` | `752 ` | `753 ` | ❌ | False |
| `473+589` | `1062` | `1060` | ❌ | True |

### Model: RNN_Reverse
| Input Question | Ground Truth | Predicted | Correct? | Has Carry? |
| :--- | :---: | :---: | :---: | :---: |
| `426+11 ` | `437 ` | `437 ` | ✅ | False |
| `211+903` | `1114` | `1114` | ✅ | True |
| `840+444` | `1284` | `1284` | ✅ | True |
| `581+111` | `692 ` | `692 ` | ✅ | False |
| `441+311` | `752 ` | `752 ` | ✅ | False |
| `473+589` | `1062` | `1062` | ✅ | True |

### Model: RNN_Peeky
| Input Question | Ground Truth | Predicted | Correct? | Has Carry? |
| :--- | :---: | :---: | :---: | :---: |
| `426+11 ` | `437 ` | `447 ` | ❌ | False |
| `211+903` | `1114` | `1114` | ✅ | True |
| `840+444` | `1284` | `1284` | ✅ | True |
| `581+111` | `692 ` | `692 ` | ✅ | False |
| `441+311` | `752 ` | `752 ` | ✅ | False |
| `473+589` | `1062` | `1062` | ✅ | True |

### Model: RNN_Reverse+Peeky
| Input Question | Ground Truth | Predicted | Correct? | Has Carry? |
| :--- | :---: | :---: | :---: | :---: |
| `426+11 ` | `437 ` | `437 ` | ✅ | False |
| `211+903` | `1114` | `1114` | ✅ | True |
| `840+444` | `1284` | `1284` | ✅ | True |
| `581+111` | `692 ` | `692 ` | ✅ | False |
| `441+311` | `752 ` | `752 ` | ✅ | False |
| `473+589` | `1062` | `1062` | ✅ | True |
