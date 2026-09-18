"""
BERT 全流程完整实现 (PyTorch 从零手写实现)
包含模块:
1. Tokenizer 基础分词与词表映射 (含 [PAD], [UNK], [CLS], [SEP], [MASK])
2. BERT 模型架构 (BertEmbeddings, MultiHeadSelfAttention, BertLayer, BertEncoder, BertPooler)
3. 预训练模块 (BertForPreTraining: MLM Head + NSP Head)
4. 预训练数据流构建 (15% 动态掩码策略: 80% MASK, 10% 随机词, 10% 保持原样; NSP 数据构造)
5. 预训练流程训练循环 (Pre-training)
6. 下游任务微调流程 (Fine-tuning for Sequence Classification)
7. 拓展机制: 如何为 BERT 增加自回归因果掩码 (UniLM 机制) 实现文本自回归生成
"""

import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

# 设置随机种子保证可复现性
torch.manual_seed(42)
random.seed(42)

# ==============================================================================
# 1. 词表与分词器 (Tokenizer)
# ==============================================================================
class SimpleTokenizer:
    """简单的字/词级 Tokenizer，包含特殊符号与编解码功能"""
    PAD_TOKEN = "[PAD]"  # 填充符
    UNK_TOKEN = "[UNK]"  # 未知词
    CLS_TOKEN = "[CLS]"  # 句首分类符
    SEP_TOKEN = "[SEP]"  # 句子分隔符
    MASK_TOKEN = "[MASK]" # 掩码占位符

    def __init__(self, vocab_list):
        self.special_tokens = [self.PAD_TOKEN, self.UNK_TOKEN, self.CLS_TOKEN, self.SEP_TOKEN, self.MASK_TOKEN]
        unique_tokens = list(dict.fromkeys(self.special_tokens + vocab_list))
        self.w2i = {w: i for i, w in enumerate(unique_tokens)}
        self.i2w = {i: w for i, w in enumerate(unique_tokens)}

    @property
    def pad_id(self): return self.w2i[self.PAD_TOKEN]
    @property
    def unk_id(self): return self.w2i[self.UNK_TOKEN]
    @property
    def cls_id(self): return self.w2i[self.CLS_TOKEN]
    @property
    def sep_id(self): return self.w2i[self.SEP_TOKEN]
    @property
    def mask_id(self): return self.w2i[self.MASK_TOKEN]
    @property
    def vocab_size(self): return len(self.w2i)

    def encode(self, text, add_special_tokens=True):
        tokens = text.lower().split()
        ids = [self.w2i.get(t, self.unk_id) for t in tokens]
        if add_special_tokens:
            ids = [self.cls_id] + ids + [self.sep_id]
        return ids

    def decode(self, ids):
        tokens = [self.i2w.get(i, self.UNK_TOKEN) for i in ids]
        return " ".join(tokens)


# ==============================================================================
# 2. BERT 模型核心架构定义
# ==============================================================================
class BertConfig:
    """BERT 超参数配置"""
    def __init__(
        self,
        vocab_size=100,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=256,
        max_position_embeddings=128,
        type_vocab_size=2,
        dropout_prob=0.1
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.dropout_prob = dropout_prob


class BertEmbeddings(nn.Module):
    """
    BERT 输入嵌入层:
    Total Embedding = Token Embedding + Position Embedding + Segment (Token Type) Embedding
    """
    def __init__(self, config: BertConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        # 生成可学习的位置索引 [0, 1, ..., seq_length - 1]
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device).unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):
    """多头自注意力机制 (Multi-Head Self-Attention)"""
    def __init__(self, config: BertConfig):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(f"hidden_size ({config.hidden_size}) must be divisible by num_attention_heads ({config.num_attention_heads})")

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = config.hidden_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.dropout_prob)

    def transpose_for_scores(self, x):
        # [batch_size, seq_len, hidden_size] -> [batch_size, num_heads, seq_len, head_size]
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None):
        mixed_query = self.query(hidden_states)
        mixed_key = self.key(hidden_states)
        mixed_value = self.value(hidden_states)

        q = self.transpose_for_scores(mixed_query)
        k = self.transpose_for_scores(mixed_key)
        v = self.transpose_for_scores(mixed_value)

        # Scaled Dot-Product: Q * K^T / sqrt(d_k)
        attention_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.attention_head_size)
        
        if attention_mask is not None:
            # attention_mask 形状为 [batch_size, 1, 1, seq_len] 或 [batch_size, 1, seq_len, seq_len]
            # 0 位置加上极大负数 -10000.0，使 softmax 之后权重接近 0
            attention_scores = attention_scores + attention_mask

        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        # Context = Attention_weights * Value
        context_layer = torch.matmul(attention_probs, v)
        # 维度复原 [batch_size, seq_len, hidden_size]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_shape)
        return context_layer


class BertSelfOutput(nn.Module):
    """残差连接与 LayerNorm"""
    def __init__(self, config: BertConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, hidden_states, attention_mask=None):
        self_output = self.self(hidden_states, attention_mask)
        attention_output = self.output(self_output, hidden_states)
        return attention_output


class BertIntermediate(nn.Module):
    """前馈全连接网络 (Feed-Forward) 第一层，激活函数使用 GELU"""
    def __init__(self, config: BertConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)

    def forward(self, hidden_states):
        return F.gelu(self.dense(hidden_states))


class BertOutput(nn.Module):
    """前馈全连接网络第二层 + 残差与 LayerNorm"""
    def __init__(self, config: BertConfig):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    """单个 Transformer Encoder Block"""
    def __init__(self, config: BertConfig):
        super().__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask=None):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertEncoder(nn.Module):
    """多层 Transformer Encoder 堆叠"""
    def __init__(self, config: BertConfig):
        super().__init__()
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask=None):
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
        return hidden_states


class BertPooler(nn.Module):
    """抽取 [CLS] 位置的特征向量并经过全连接 + Tanh 作为整句表征"""
    def __init__(self, config: BertConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # 取每个样本的第一个 token ([CLS]) 的隐藏状态
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertModel(nn.Module):
    """基础 BERT 模型 (Backbone)"""
    def __init__(self, config: BertConfig):
        super().__init__()
        self.config = config
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

    def get_extended_attention_mask(self, attention_mask):
        """将 2D 的 padding mask [batch, seq_len] 扩展成注意力矩阵偏置 [batch, 1, 1, seq_len]"""
        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        elif attention_mask.dim() == 3:
            extended_attention_mask = attention_mask.unsqueeze(1)
        else:
            extended_attention_mask = attention_mask
        
        # 将 1 (有效位置) 映射为 0.0，0 (填充位置) 映射为 -10000.0
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        # 转换 attention mask 格式
        extended_attention_mask = self.get_extended_attention_mask(attention_mask)
        embedding_output = self.embeddings(input_ids, token_type_ids)
        sequence_output = self.encoder(embedding_output, extended_attention_mask)
        pooled_output = self.pooler(sequence_output)
        return sequence_output, pooled_output


# ==============================================================================
# 3. 预训练任务头 (BertForPreTraining)
# ==============================================================================
class BertLMPredictionHead(nn.Module):
    """Masked Language Model (MLM) 预测头"""
    def __init__(self, config: BertConfig, word_embeddings_weight):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # 权重共享 (Weight Tying): 输出解码矩阵与输入词嵌入矩阵共享参数
        self.decoder.weight = word_embeddings_weight
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        x = self.dense(hidden_states)
        x = F.gelu(x)
        x = self.layer_norm(x)
        x = self.decoder(x)
        return x


class BertForPreTraining(nn.Module):
    """包含 MLM 和 NSP (Next Sentence Prediction) 双任务的预训练模型"""
    def __init__(self, config: BertConfig):
        super().__init__()
        self.bert = BertModel(config)
        self.mlm_head = BertLMPredictionHead(config, self.bert.embeddings.word_embeddings.weight)
        self.nsp_head = nn.Linear(config.hidden_size, 2) # 二分类: 0 (IsNext), 1 (NotNext)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None, next_sentence_label=None):
        sequence_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
        
        prediction_scores = self.mlm_head(sequence_output) # [batch, seq_len, vocab_size]
        seq_relationship_score = self.nsp_head(pooled_output) # [batch, 2]

        total_loss = None
        if masked_lm_labels is not None and next_sentence_label is not None:
            # MLM 损失: 忽略标签为 -100 的位置 (即未被 mask 的 token)
            loss_fct_mlm = nn.CrossEntropyLoss(ignore_index=-100)
            mlm_loss = loss_fct_mlm(prediction_scores.view(-1, self.bert.config.vocab_size), masked_lm_labels.view(-1))

            # NSP 损失
            loss_fct_nsp = nn.CrossEntropyLoss()
            nsp_loss = loss_fct_nsp(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))

            total_loss = mlm_loss + nsp_loss
            return total_loss, prediction_scores, seq_relationship_score
        
        return prediction_scores, seq_relationship_score


# ==============================================================================
# 4. 预训练数据流构建 (MLM 掩码规则 + NSP 句对拼接)
# ==============================================================================
def create_pretraining_sample(sentence_a, sentence_b, is_next, tokenizer: SimpleTokenizer, max_len=16):
    """
    实现论文标准的 15% Mask 策略:
    - 80% 替换为 [MASK]
    - 10% 随机替换为词表中其它 token
    - 10% 保持不变
    """
    tokens_a = tokenizer.encode(sentence_a, add_special_tokens=False)
    tokens_b = tokenizer.encode(sentence_b, add_special_tokens=False)

    # 组合为 [CLS] + A + [SEP] + B + [SEP]
    tokens = [tokenizer.cls_id] + tokens_a + [tokenizer.sep_id] + tokens_b + [tokenizer.sep_id]
    token_type_ids = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)
    
    # 构造 MLM 标签，初始全部为 -100 (不计入损失)
    masked_labels = [-100] * len(tokens)
    
    # 候选掩码位置: 排除 [CLS] 和 [SEP]
    candidate_indices = [
        i for i, token_id in enumerate(tokens) 
        if token_id not in (tokenizer.cls_id, tokenizer.sep_id)
    ]
    random.shuffle(candidate_indices)
    num_to_mask = max(1, int(round(len(candidate_indices) * 0.15)))

    for idx in candidate_indices[:num_to_mask]:
        masked_labels[idx] = tokens[idx] # 目标预测真实 token
        prob = random.random()
        if prob < 0.8:
            tokens[idx] = tokenizer.mask_id # 80%: 替换为 [MASK]
        elif prob < 0.9:
            tokens[idx] = random.randint(5, tokenizer.vocab_size - 1) # 10%: 随机词
        else:
            pass # 10%: 保持原样

    # 截断或填充至固定长度 max_len
    if len(tokens) > max_len:
        tokens = tokens[:max_len]
        token_type_ids = token_type_ids[:max_len]
        masked_labels = masked_labels[:max_len]

    attention_mask = [1] * len(tokens)
    pad_len = max_len - len(tokens)
    if pad_len > 0:
        tokens += [tokenizer.pad_id] * pad_len
        token_type_ids += [0] * pad_len
        attention_mask += [0] * pad_len
        masked_labels += [-100] * pad_len

    return {
        "input_ids": torch.tensor(tokens, dtype=torch.long),
        "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "masked_lm_labels": torch.tensor(masked_labels, dtype=torch.long),
        "next_sentence_label": torch.tensor(0 if is_next else 1, dtype=torch.long)
    }


# ==============================================================================
# 5. 下游微调模块: 文本分类 (BertForSequenceClassification)
# ==============================================================================
class BertForSequenceClassification(nn.Module):
    """下游微调模型: 取 [CLS] 表征接分类器"""
    def __init__(self, bert_model: BertModel, num_classes=2):
        super().__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(bert_model.config.dropout_prob)
        self.classifier = nn.Linear(bert_model.config.hidden_size, num_classes)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            return loss, logits
        return logits


# ==============================================================================
# 6. 拓展实现: 如何让 BERT 具备自回归生成能力 (UniLM Seq2Seq 因果掩码机制)
# ==============================================================================
class BertAutoregressiveGenerator:
    """
    通过巧妙设计 Attention Mask 将 BERT 转化为自回归生成器 (参考微软 UniLM 思想)
    掩码矩阵逻辑:
    - Source (输入 Prompt): 双向注意力 (互相可见)
    - Target (生成序列): 下三角因果注意力 (只能看到之前的 Token 以及全部 Source)
    """
    def __init__(self, bert_model: BertModel, tokenizer: SimpleTokenizer):
        self.bert = bert_model
        self.tokenizer = tokenizer
        # 解码头复用输入词向量权重
        self.lm_head = nn.Linear(bert_model.config.hidden_size, tokenizer.vocab_size, bias=False)
        self.lm_head.weight = bert_model.embeddings.word_embeddings.weight

    def build_unilm_mask(self, prompt_len, target_len):
        """
        构建 UniLM Seq2Seq 注意力掩码
        Prompt 之间为全 1 (双向)，Target 内部为下三角矩阵 (因果单向)
        Target 对 Prompt 全为 1，Prompt 对 Target 全为 0 (防止信息从后往前泄漏)
        """
        total_len = prompt_len + target_len
        # 基础全 0 矩阵
        mask = torch.zeros((total_len, total_len), dtype=torch.float32)
        # 1. Prompt 区域互相可见
        mask[:prompt_len, :prompt_len] = 1.0
        # 2. Target 区域可以看到所有 Prompt
        mask[prompt_len:, :prompt_len] = 1.0
        # 3. Target 区域内部自回归下三角可见
        target_causal = torch.tril(torch.ones((target_len, target_len), dtype=torch.float32))
        mask[prompt_len:, prompt_len:] = target_causal
        return mask

    def generate(self, prompt_text, max_gen_len=5):
        self.bert.eval()
        prompt_tokens = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        prompt_tokens = [self.tokenizer.cls_id] + prompt_tokens
        
        generated = []
        for _ in range(max_gen_len):
            current_seq = prompt_tokens + generated
            p_len = len(prompt_tokens)
            t_len = len(generated)

            # 构建当前步的注意力和序列
            input_ids = torch.tensor([current_seq], dtype=torch.long)
            if t_len == 0:
                # 初始只有 prompt，全部双向
                attn_mask = torch.ones((1, 1, p_len, p_len))
            else:
                unilm_mask = self.build_unilm_mask(p_len, t_len)
                attn_mask = unilm_mask.unsqueeze(0).unsqueeze(0) # [1, 1, seq_len, seq_len]

            # 扩展掩码: 1 -> 0.0, 0 -> -10000.0
            extended_mask = (1.0 - attn_mask) * -10000.0
            
            with torch.no_grad():
                emb = self.bert.embeddings(input_ids)
                hidden = self.bert.encoder(emb, extended_mask)
                last_token_hidden = hidden[:, -1, :] # 获取当前待预测位置隐藏状态
                logits = self.lm_head(last_token_hidden)
                next_token_id = torch.argmax(logits, dim=-1).item()

            if next_token_id == self.tokenizer.sep_id:
                break
            generated.append(next_token_id)

        return self.tokenizer.decode(generated)


# ==============================================================================
# 7. 全流程端到端演示 (测试运行)
# ==============================================================================
def run_bert_pipeline_demo():
    print("=" * 60)
    print(" 1. 初始化词表与配置")
    print("=" * 60)
    corpus = [
        "deep learning and natural language processing are fascinating",
        "language models understand semantics through self attention",
        "transformers have revolutionized artificial intelligence and text representation",
        "machine learning algorithms analyze large scale data efficiently"
    ]
    vocab = []
    for s in corpus:
        vocab.extend(s.split())
    tokenizer = SimpleTokenizer(vocab)
    print(f"词表总大小 (含特殊符号): {tokenizer.vocab_size}")

    config = BertConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=32
    )

    print("\n" + "=" * 60)
    print(" 2. 预训练阶段 (Pre-training: MLM + NSP)")
    print("=" * 60)
    pretrain_model = BertForPreTraining(config)
    optimizer = torch.optim.Adam(pretrain_model.parameters(), lr=1e-3)

    # 构造预训练批次样本
    batch_samples = [
        create_pretraining_sample(
            "deep learning and natural language",
            "processing are fascinating",
            is_next=True,
            tokenizer=tokenizer,
            max_len=16
        ),
        create_pretraining_sample(
            "deep learning and natural language",
            "machine learning algorithms analyze data",
            is_next=False,
            tokenizer=tokenizer,
            max_len=16
        )
    ]
    
    input_ids = torch.stack([s["input_ids"] for s in batch_samples])
    token_type_ids = torch.stack([s["token_type_ids"] for s in batch_samples])
    attention_mask = torch.stack([s["attention_mask"] for s in batch_samples])
    masked_lm_labels = torch.stack([s["masked_lm_labels"] for s in batch_samples])
    next_sentence_label = torch.stack([s["next_sentence_label"] for s in batch_samples])

    print("开始执行预训练一步梯度更新...")
    pretrain_model.train()
    optimizer.zero_grad()
    loss, mlm_logits, nsp_logits = pretrain_model(
        input_ids=input_ids,
        token_type_ids=token_type_ids,
        attention_mask=attention_mask,
        masked_lm_labels=masked_lm_labels,
        next_sentence_label=next_sentence_label
    )
    loss.backward()
    optimizer.step()
    print(f"预训练前向与反向传播成功! Total Loss: {loss.item():.4f}")

    print("\n" + "=" * 60)
    print(" 3. 下游任务微调 (Fine-tuning: 文本分类)")
    print("=" * 60)
    # 取出预训练好的骨干网络 BertModel 进行微调
    classifier_model = BertForSequenceClassification(pretrain_model.bert, num_classes=2)
    cls_optimizer = torch.optim.Adam(classifier_model.parameters(), lr=5e-4)

    # 构造假拟分类数据: 正类 1, 负类 0
    text_sample = "deep learning is fascinating"
    sample_ids = torch.tensor([tokenizer.encode(text_sample)], dtype=torch.long)
    sample_label = torch.tensor([1], dtype=torch.long)

    classifier_model.train()
    cls_optimizer.zero_grad()
    cls_loss, logits = classifier_model(sample_ids, labels=sample_label)
    cls_loss.backward()
    cls_optimizer.step()
    print(f"下游文本分类微调成功! 样本分类损失 Loss: {cls_loss.item():.4f}, 预测 Logits: {logits.detach().numpy()}")

    print("\n" + "=" * 60)
    print(" 4. 扩展探索: BERT + 因果掩码 (自回归文本解码演示)")
    print("=" * 60)
    generator = BertAutoregressiveGenerator(pretrain_model.bert, tokenizer)
    prompt = "deep learning"
    gen_result = generator.generate(prompt, max_gen_len=4)
    print(f"输入 Prompt: '{prompt}'")
    print(f"自回归逐步解码生成的 Token 序列: '{gen_result}'")
    print("=" * 60)


if __name__ == "__main__":
    run_bert_pipeline_demo()
