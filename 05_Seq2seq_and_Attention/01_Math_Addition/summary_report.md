# Seq2Seq Optimization Experiment Summary (LSTM on Arithmetic Addition)

| Model | Params | Training Time (s) | Final Loss | Final EM Acc (%) | Carry Acc (%) | No-Carry Acc (%) | Reached 50% Epoch | Reached 90% Epoch |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | 151,597 | 76.5 | 0.6000 | **26.26%** | 27.25% | 21.28% | - | - |
| **Reverse** | 151,597 | 60.1 | 0.5614 | **23.22%** | 23.01% | 24.30% | - | - |
| **Peeky** | 218,797 | 63.9 | 0.0320 | **97.06%** | 97.51% | 94.80% | 11 | 17 |
| **Reverse+Peeky** | 218,797 | 94.2 | 0.2409 | **66.80%** | 68.03% | 60.58% | 20 | - |

## Sample Predictions & Qualitative Error Analysis

### Model: Baseline
| Input Question | Ground Truth | Predicted | Correct? | Has Carry? | Digits |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `426+11 ` | `437 ` | `434 ` | ❌ | False | 3 |
| `211+903` | `1114` | `1116` | ❌ | True | 3 |
| `840+444` | `1284` | `1283` | ❌ | True | 3 |
| `581+111` | `692 ` | `693 ` | ❌ | False | 3 |
| `441+311` | `752 ` | `743 ` | ❌ | False | 3 |
| `473+589` | `1062` | `1062` | ✅ | True | 3 |

### Model: Reverse
| Input Question | Ground Truth | Predicted | Correct? | Has Carry? | Digits |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `426+11 ` | `437 ` | `444 ` | ❌ | False | 3 |
| `211+903` | `1114` | `1104` | ❌ | True | 3 |
| `840+444` | `1284` | `1274` | ❌ | True | 3 |
| `581+111` | `692 ` | `692 ` | ✅ | False | 3 |
| `441+311` | `752 ` | `742 ` | ❌ | False | 3 |
| `473+589` | `1062` | `1053` | ❌ | True | 3 |

### Model: Peeky
| Input Question | Ground Truth | Predicted | Correct? | Has Carry? | Digits |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `426+11 ` | `437 ` | `437 ` | ✅ | False | 3 |
| `211+903` | `1114` | `1114` | ✅ | True | 3 |
| `840+444` | `1284` | `1284` | ✅ | True | 3 |
| `581+111` | `692 ` | `692 ` | ✅ | False | 3 |
| `441+311` | `752 ` | `752 ` | ✅ | False | 3 |
| `473+589` | `1062` | `1062` | ✅ | True | 3 |

### Model: Reverse+Peeky
| Input Question | Ground Truth | Predicted | Correct? | Has Carry? | Digits |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `426+11 ` | `437 ` | `436 ` | ❌ | False | 3 |
| `211+903` | `1114` | `1113` | ❌ | True | 3 |
| `840+444` | `1284` | `1284` | ✅ | True | 3 |
| `581+111` | `692 ` | `691 ` | ❌ | False | 3 |
| `441+311` | `752 ` | `751 ` | ❌ | False | 3 |
| `473+589` | `1062` | `1062` | ✅ | True | 3 |
