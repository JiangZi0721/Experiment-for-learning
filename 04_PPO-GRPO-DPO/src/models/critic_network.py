import torch
import torch.nn as nn
from typing import Optional, List

class WhiteBoxCriticNetwork(nn.Module):
    """
    白盒可透视价值网络 (Critic)
    结合时序语义感知先验与可微神经价值头，对序列每一个 Token 预测状态价值 V(s_t)
    真实反映时序价值演进（称赞/礼貌得正分，失控/违规断崖跳水）
    """
    def __init__(self, vocab_size: int = 512, embed_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True, num_layers=2)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, input_ids: torch.Tensor, token_texts: Optional[List[str]] = None) -> torch.Tensor:
        """
        计算序列每一个 Token 之后的状态价值 V(s_t)
        完全由可微神经网络计算，杜绝任何硬编码规则干预！
        Returns:
            values: [Batch_Size, Seq_Len] 预估价值 (实数心电图)
        """
        emb = self.embedding(input_ids)
        out, _ = self.rnn(emb)
        raw_val = self.value_head(out).squeeze(-1) # [Batch_Size, Seq_Len]
        return raw_val
