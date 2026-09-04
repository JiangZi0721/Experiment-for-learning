import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional

class ToyTokenizer:
    """教学用白盒分词器 (支持中文字符、英文字符与结构化 CoT 标签)"""
    def __init__(self, vocab_size: int = 2048):
        self.pad_token = "<pad>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.think_start = "<think>"
        self.think_end = "</think>"
        self.ans_start = "<answer>"
        self.ans_end = "</answer>"

        # 基础特殊词表
        special_tokens = [
            self.pad_token, self.bos_token, self.eos_token,
            self.think_start, self.think_end, self.ans_start, self.ans_end
        ]
        self.w2i: Dict[str, int] = {tok: idx for idx, tok in enumerate(special_tokens)}
        self.i2w: Dict[int, str] = {idx: tok for idx, tok in enumerate(special_tokens)}
        self.max_vocab = vocab_size

    def build_vocab_from_texts(self, texts: List[str]):
        """根据输入语料动态扩充词表"""
        for text in texts:
            for ch in text:
                if ch not in self.w2i and len(self.w2i) < self.max_vocab - 1:
                    idx = len(self.w2i)
                    self.w2i[ch] = idx
                    self.i2w[idx] = ch

    def encode(self, text: str, add_bos: bool = True) -> List[int]:
        """将文本编码为 Token ID 序列"""
        tokens = []
        if add_bos:
            tokens.append(self.w2i[self.bos_token])

        i = 0
        while i < len(text):
            matched_special = False
            for special in [self.think_start, self.think_end, self.ans_start, self.ans_end]:
                if text[i:].startswith(special):
                    if special not in self.w2i:
                        idx = len(self.w2i)
                        self.w2i[special] = idx
                        self.i2w[idx] = special
                    tokens.append(self.w2i[special])
                    i += len(special)
                    matched_special = True
                    break
            if not matched_special:
                ch = text[i]
                if ch not in self.w2i:
                    idx = len(self.w2i)
                    self.w2i[ch] = idx
                    self.i2w[idx] = ch
                tokens.append(self.w2i[ch])
                i += 1
        return tokens

    def decode(self, token_ids: List[int]) -> str:
        """将 Token ID 序列还原为文本"""
        res = []
        for tid in token_ids:
            tok = self.i2w.get(tid, "")
            if tok in [self.pad_token, self.bos_token, self.eos_token]:
                continue
            res.append(tok)
        return "".join(res)

    @property
    def vocab_size(self) -> int:
        return max(len(self.w2i) + 16, 512)


class WhiteBoxPolicyNetwork(nn.Module):
    """
    白盒可透视策略网络 (Actor)
    采用因果自回归架构 (Embedding -> GRU -> LM Head)
    对外接口与标准 CausalLM 完全对齐，提供对数似然评估、采样生成与动作熵计算
    """
    def __init__(self, vocab_size: int = 512, embed_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True, num_layers=2)
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        前向传播计算 logits
        input_ids: [Batch_Size, Seq_Len]
        """
        emb = self.embedding(input_ids)
        out, _ = self.rnn(emb)
        logits = self.lm_head(out)
        return logits

    def evaluate_actions(self, input_ids: torch.Tensor, action_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        评估动作序列的 log 概率以及策略熵 (Entropy)
        Returns:
            action_log_probs: [Batch_Size, Seq_Len-1]
            entropy: 标量动作熵
        """
        logits = self.forward(input_ids)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        log_probs = F.log_softmax(shift_logits, dim=-1)
        action_log_probs = torch.gather(log_probs, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)

        probs = F.softmax(shift_logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)

        if action_mask is not None:
            action_log_probs = action_log_probs * action_mask
            entropy = (entropy * action_mask).sum() / (action_mask.sum() + 1e-8)
        else:
            entropy = entropy.mean()

        return action_log_probs, entropy

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 16,
        temperature: float = 1.0,
        top_k: int = 10,
        repetition_penalty: float = 1.1,
        eos_token_id: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        cur_ids = prompt_ids.clone()
        gen_log_probs_list = []

        for _ in range(max_new_tokens):
            logits = self.forward(cur_ids)[:, -1, :].clone()

            if repetition_penalty != 1.0:
                for b in range(cur_ids.size(0)):
                    for prev_token in set(cur_ids[b].tolist()):
                        if logits[b, prev_token] < 0:
                            logits[b, prev_token] *= repetition_penalty
                        else:
                            logits[b, prev_token] /= repetition_penalty

            logits = logits / max(temperature, 1e-4)

            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            next_token = dist.sample().unsqueeze(-1)
            token_log_prob = dist.log_prob(next_token.squeeze(-1)).unsqueeze(-1)

            cur_ids = torch.cat([cur_ids, next_token], dim=1)
            gen_log_probs_list.append(token_log_prob)

            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        gen_log_probs = torch.cat(gen_log_probs_list, dim=1) if gen_log_probs_list else torch.empty((prompt_ids.size(0), 0))
        return cur_ids, gen_log_probs


def load_base_policy_and_tokenizer(checkpoint_path: Optional[str] = None) -> Tuple[WhiteBoxPolicyNetwork, ToyTokenizer]:
    """加载已就绪的基座策略模型与词表"""
    from pathlib import Path
    if checkpoint_path is None:
        p = Path(__file__).resolve().parent.parent.parent / "data" / "base_policy.pt"
    else:
        p = Path(checkpoint_path)

    tokenizer = ToyTokenizer()
    if p.exists():
        ckpt = torch.load(p, map_location="cpu")
        tokenizer.w2i = ckpt["w2i"]
        tokenizer.i2w = ckpt["i2w"]
        policy = WhiteBoxPolicyNetwork(vocab_size=tokenizer.vocab_size)
        policy.load_state_dict(ckpt["state_dict"])
    else:
        policy = WhiteBoxPolicyNetwork(vocab_size=tokenizer.vocab_size)
    return policy, tokenizer
