# -*- coding: utf-8 -*-
"""
Dataset generator for Sequence Labeling & Label Bias Simulation
1. 命名实体识别 (NER) 句法依赖仿真数据集
2. 经典 McCallum & Lafferty 标注偏置有限状态机 (Label Bias Graph)
"""
import random
import torch
import numpy as np

TAGS = ["O", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG"]
START_TAG = "<START>"
STOP_TAG = "<STOP>"
PAD_TAG = "<PAD>"

ALL_TAGS = TAGS + [START_TAG, STOP_TAG, PAD_TAG]
TAG2ID = {t: i for i, t in enumerate(ALL_TAGS)}
ID2TAG = {i: t for i, t in enumerate(ALL_TAGS)}

START_TAG_IDX = TAG2ID[START_TAG]
STOP_TAG_IDX = TAG2ID[STOP_TAG]
PAD_TAG_IDX = TAG2ID[PAD_TAG]

# 常见词汇库
FIRST_NAMES = ["张", "李", "王", "赵", "陈", "刘", "周", "杨"]
LAST_NAMES = ["伟", "芳", "敏", "静", "强", "军", "洋", "艳", "明"]
CITIES = ["北京", "上海", "广州", "深圳", "杭州", "南京", "成都", "武汉"]
ORGS = ["腾讯", "阿里", "百度", "字节", "美团", "京东", "华为", "小米"]
VERBS = ["前往", "访问", "参观", "离开", "投资", "考察", "入住", "成立", "签约"]
NOUNS = ["总部", "分部", "园区", "会议", "中心", "大楼", "实验室", "大学", "论坛"]
CONNECTORS = ["并在", "然后", "随后", "同", "与", "在", "计划", "决定"]

def is_transition_valid(prev_tag_str, cur_tag_str):
    """
    语法硬约束校验函数：
    判断从 prev_tag_str 到 cur_tag_str 是否符合命名实体语法拓扑：
    - I-PER 只能紧随 B-PER 或 I-PER
    - I-LOC 只能紧随 B-LOC 或 I-LOC
    - I-ORG 只能紧随 B-ORG 或 I-ORG
    - 其他组合均为非法语法跳变
    """
    if cur_tag_str.startswith("I-"):
        ent_type = cur_tag_str.split("-")[1]
        valid_prev = [f"B-{ent_type}", f"I-{ent_type}"]
        return prev_tag_str in valid_prev
    return True

def generate_ner_sentence():
    """
    生成符合生物学/语言学真实命名实体规范的单条数据
    """
    p_len = random.choice([2, 3])
    per = random.choice(FIRST_NAMES) + "".join(random.choices(LAST_NAMES, k=p_len - 1))
    loc = random.choice(CITIES)
    org = random.choice(ORGS)
    v1 = random.choice(VERBS)
    v2 = random.choice(VERBS)
    c = random.choice(CONNECTORS)
    n = random.choice(NOUNS)

    # 随机选择一种句型拓扑
    template = random.choice([1, 2, 3])
    tokens = []
    tags = []

    if template == 1:
        # [PER] v1 [LOC] c v2 [ORG] n
        tokens.extend(list(per))
        tags.append("B-PER")
        tags.extend(["I-PER"] * (len(per) - 1))

        tokens.extend(list(v1))
        tags.extend(["O"] * len(v1))

        tokens.extend(list(loc))
        tags.append("B-LOC")
        tags.extend(["I-LOC"] * (len(loc) - 1))

        tokens.extend(list(c))
        tags.extend(["O"] * len(c))

        tokens.extend(list(v2))
        tags.extend(["O"] * len(v2))

        tokens.extend(list(org))
        tags.append("B-ORG")
        tags.extend(["I-ORG"] * (len(org) - 1))

        tokens.extend(list(n))
        tags.extend(["O"] * len(n))
    elif template == 2:
        # 在 [LOC] 的 [ORG] 与 [PER] 举行 n
        tokens.extend(list("在"))
        tags.append("O")

        tokens.extend(list(loc))
        tags.append("B-LOC")
        tags.extend(["I-LOC"] * (len(loc) - 1))

        tokens.extend(list("的"))
        tags.append("O")

        tokens.extend(list(org))
        tags.append("B-ORG")
        tags.extend(["I-ORG"] * (len(org) - 1))

        tokens.extend(list("与"))
        tags.append("O")

        tokens.extend(list(per))
        tags.append("B-PER")
        tags.extend(["I-PER"] * (len(per) - 1))

        tokens.extend(list("举行"))
        tags.extend(["O", "O"])

        tokens.extend(list(n))
        tags.extend(["O"] * len(n))
    else:
        # [ORG] 位于 [LOC] 由 [PER] 负责
        tokens.extend(list(org))
        tags.append("B-ORG")
        tags.extend(["I-ORG"] * (len(org) - 1))

        tokens.extend(list("位于"))
        tags.extend(["O", "O"])

        tokens.extend(list(loc))
        tags.append("B-LOC")
        tags.extend(["I-LOC"] * (len(loc) - 1))

        tokens.extend(list("由"))
        tags.append("O")

        tokens.extend(list(per))
        tags.append("B-PER")
        tags.extend(["I-PER"] * (len(per) - 1))

        tokens.extend(list("负责"))
        tags.extend(["O", "O"])

    return tokens, tags

def build_ner_dataset(num_samples=1500, test_ratio=0.2, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    raw_data = []
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for _ in range(num_samples):
        tokens, tags = generate_ner_sentence()
        for tok in tokens:
            if tok not in vocab:
                vocab[tok] = len(vocab)
        raw_data.append((tokens, tags))

    # 切分训练集与测试集
    num_test = int(num_samples * test_ratio)
    test_data = raw_data[:num_test]
    train_data = raw_data[num_test:]

    return train_data, test_data, vocab

def collate_fn_ner(batch, vocab):
    """DataLoader 对齐 Padding 批处理函数"""
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    max_len = max(len(x[0]) for x in batch)
    batch_size = len(batch)

    x_tensor = torch.full((max_len, batch_size), vocab["<PAD>"], dtype=torch.long)
    y_tensor = torch.full((max_len, batch_size), PAD_TAG_IDX, dtype=torch.long)
    mask_tensor = torch.zeros((max_len, batch_size), dtype=torch.bool)

    for i, (tokens, tags) in enumerate(batch):
        seq_len = len(tokens)
        token_ids = [vocab.get(tok, vocab["<UNK>"]) for tok in tokens]
        tag_ids = [TAG2ID[t] for t in tags]

        x_tensor[:seq_len, i] = torch.tensor(token_ids, dtype=torch.long)
        y_tensor[:seq_len, i] = torch.tensor(tag_ids, dtype=torch.long)
        mask_tensor[:seq_len, i] = True

    return x_tensor, y_tensor, mask_tensor
