# Machine Translation Experiment Summary (English -> French)

| Model | Params | Training Time (s) | Final Val Loss | Final BLEU | Short BLEU (<=5) | Long BLEU (>=6) | Exact Match (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | 1,032,088 | 96.0 | 2.4449 | **15.15** | 17.86 | 13.74 | 0.93% |
| **Reverse** | 1,032,088 | 99.6 | 2.4030 | **15.65** | 17.55 | 14.45 | 0.60% |
| **Peeky** | 1,534,872 | 118.0 | 2.4742 | **18.05** | 19.05 | 17.22 | 0.93% |
| **Reverse+Peeky** | 1,534,872 | 108.5 | 2.4670 | **18.97** | 19.26 | 18.29 | 1.07% |

## Sample Translation Predictions

### Model: Baseline
| Source (English) | Reference (French) | Prediction (Hypothesis) | Exact Match? |
| :--- | :--- | :--- | :---: |
| `i m paying .` | `je suis en train de payer .` | `je suis surpris .` | ❌ |
| `i use firefox .` | `j utilise firefox .` | `je me <unk> <unk> .` | ❌ |
| `do you want to see more ?` | `voulez vous en voir davantage ?` | `veux tu aller avec nous ?` | ❌ |
| `the bad weather affected his health .` | `le mauvais temps affecta sa sante .` | `le <unk> <unk> commence a vendre .` | ❌ |
| `it s safe here .` | `on est en securite ici .` | `c est le mien .` | ❌ |
| `everyone is staring at tom .` | `tout le monde devisage tom .` | `tout le monde <unk> la tele de tom .` | ❌ |

### Model: Reverse
| Source (English) | Reference (French) | Prediction (Hypothesis) | Exact Match? |
| :--- | :--- | :--- | :---: |
| `. paying m i` | `je suis en train de payer .` | `je suis <unk> .` | ❌ |
| `. firefox use i` | `j utilise firefox .` | `je me sens toujours <unk> .` | ❌ |
| `? more see to want you do` | `voulez vous en voir davantage ?` | `veux tu que je fasse ?` | ❌ |
| `. health his affected weather bad the` | `le mauvais temps affecta sa sante .` | `le <unk> a tue le <unk> .` | ❌ |
| `. here safe s it` | `on est en securite ici .` | `il est ici .` | ❌ |
| `. tom at staring is everyone` | `tout le monde devisage tom .` | `tout le monde est <unk> de tom .` | ❌ |

### Model: Peeky
| Source (English) | Reference (French) | Prediction (Hypothesis) | Exact Match? |
| :--- | :--- | :--- | :---: |
| `i m paying .` | `je suis en train de payer .` | `je suis totalement en train .` | ❌ |
| `i use firefox .` | `j utilise firefox .` | `je me sens a utiliser .` | ❌ |
| `do you want to see more ?` | `voulez vous en voir davantage ?` | `voulez vous dans le train ?` | ❌ |
| `the bad weather affected his health .` | `le mauvais temps affecta sa sante .` | `le temps s est <unk> comme une <unk> .` | ❌ |
| `it s safe here .` | `on est en securite ici .` | `il est ici que c est ici .` | ❌ |
| `everyone is staring at tom .` | `tout le monde devisage tom .` | `tout le monde est en colere par la .` | ❌ |

### Model: Reverse+Peeky
| Source (English) | Reference (French) | Prediction (Hypothesis) | Exact Match? |
| :--- | :--- | :--- | :---: |
| `. paying m i` | `je suis en train de payer .` | `je suis <unk> .` | ❌ |
| `. firefox use i` | `j utilise firefox .` | `je ne fais pas la <unk> .` | ❌ |
| `? more see to want you do` | `voulez vous en voir davantage ?` | `veux tu que tom soit ?` | ❌ |
| `. health his affected weather bad the` | `le mauvais temps affecta sa sante .` | `les <unk> <unk> de ses parents .` | ❌ |
| `. here safe s it` | `on est en securite ici .` | `c est ici .` | ❌ |
| `. tom at staring is everyone` | `tout le monde devisage tom .` | `tout le monde est parle en colere .` | ❌ |
