# -*- coding: utf-8 -*-
"""
RNNLM: 循环神经网络语言模型 (Recurrent Neural Network Language Model)
端到端白盒实现: TimeEmbedding -> TimeRNN -> TimeAffine -> TimeSoftmaxWithLoss
支持自回归序列生成 (Autoregressive Generation)、跨批次状态接力与微观全景探针。
"""
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from src.layers import TimeEmbedding, TimeAffine, TimeSoftmaxWithLoss
from src.time_rnn import TimeRNN
from src.gated_rnn import TimeGRU, GRUCell


class RNNLM:
    """
    基于循环神经网络的因果语言模型 (RNNLM)
    支持两种循环骨架:
      - "vanilla": 基础单步循环神经网络 (Vanilla RNN)
      - "gru":     门控循环神经网络 (Gated Recurrent Unit, GRU)

    架构链路:
        xs (N, T)
          │
          ▼ [TimeEmbedding]
        (N, T, D)
          │
          ▼ [TimeRNN / TimeGRU (stateful=True)] ◄─── h_prev (跨 Chunk 状态接力)
        (N, T, H)
          │
          ▼ [TimeAffine]
        (N, T, V)
          │
          ▼ [TimeSoftmaxWithLoss] ◄─── ts (N, T)
        Loss (标量)
    """
    def __init__(
        self,
        vocab_size: int = 1000,
        wordvec_size: int = 64,
        hidden_size: int = 128,
        init_std: float = 0.01,
        rnn_type: str = "vanilla",
        seed: Optional[int] = 42
    ):
        if seed is not None:
            np.random.seed(seed)

        V, D, H = vocab_size, wordvec_size, hidden_size
        self.vocab_size = V
        self.wordvec_size = D
        self.hidden_size = H
        self.rnn_type = rnn_type.lower()

        # 1. 初始化权重与偏置
        embed_W = (np.random.randn(V, D) * init_std).astype(np.float32)

        if self.rnn_type == "gru":
            # 门控循环网络: 3 倍隐藏维度 [reset_gate, update_gate, candidate]
            rnn_Wx = (np.random.randn(D, 3 * H) / np.sqrt(D)).astype(np.float32)
            rnn_Wh = (np.random.randn(H, 3 * H) / np.sqrt(H)).astype(np.float32)
            rnn_b = np.zeros(3 * H, dtype=np.float32)
            rnn_layer = TimeGRU(rnn_Wx, rnn_Wh, rnn_b, stateful=True)
        else:
            # 基础 Vanilla RNN
            rnn_Wx = (np.random.randn(D, H) / np.sqrt(D)).astype(np.float32)
            rnn_Wh = (np.random.randn(H, H) / np.sqrt(H)).astype(np.float32)
            rnn_b = np.zeros(H, dtype=np.float32)
            rnn_layer = TimeRNN(rnn_Wx, rnn_Wh, rnn_b, stateful=True)

        affine_W = (np.random.randn(H, V) / np.sqrt(H)).astype(np.float32)
        affine_b = np.zeros(V, dtype=np.float32)

        # 2. 组装网络层 (注意: 默认开启 stateful=True 以支持截断跨块状态延续)
        self.layers = [
            TimeEmbedding(embed_W),
            rnn_layer,
            TimeAffine(affine_W, affine_b)
        ]
        self.loss_layer = TimeSoftmaxWithLoss()
        self.rnn_layer = self.layers[1]

        # 3. 集中管理所有可训练参数与梯度指针
        self.params: List[np.ndarray] = []
        self.grads: List[np.ndarray] = []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def reset_state(self):
        """重置 RNN 时序隐藏状态 (在序列开头或不同文档间重置)"""
        self.rnn_layer.reset_state()

    def set_state(self, h: np.ndarray):
        """设置指定的隐藏状态"""
        self.rnn_layer.set_state(h)

    def get_state(self) -> Optional[np.ndarray]:
        """获取当前的隐藏状态"""
        return self.rnn_layer.h.copy() if self.rnn_layer.h is not None else None

    def forward(self, xs: np.ndarray, ts: np.ndarray) -> float:
        """
        前向计算完整图: 输入序列 -> 逐层传递 -> 计算时序交叉熵损失

        参数:
            xs: 输入 Token 矩阵, shape=(N, T)
            ts: 目标 Token 矩阵, shape=(N, T) (通常为 xs 向右偏移 1 个位置的下一个词)
        """
        self.current_xs = xs
        for layer in self.layers:
            xs = layer.forward(xs)
        loss = self.loss_layer.forward(xs, ts)
        return loss

    def backward(self, dout: float = 1.0) -> float:
        """
        时序反向传播: 穿透 Softmax -> Affine -> TimeRNN (BPTT) -> Embedding
        """
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
            if isinstance(dout, tuple):
                dout = dout[0]  # 提取主时序分支梯度 dxs, 忽略跨块断开的 dh
        return dout

    def predict_next_token_logits(self, x_token: int, h_prev: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        单步预测下一个 Token 的未归一化对数概率 Logits 与更新后的隐藏状态

        参数:
            x_token: 单个词/字符的索引 (int)
            h_prev: 上一步的隐藏状态 (1, H), 若为 None 则使用内部状态或全 0
        返回:
            (logits, h_new): shape=(V,), shape=(1, H)
        """
        N = 1
        if h_prev is None:
            if self.rnn_layer.h is not None:
                h_prev = self.rnn_layer.h[:1]
            else:
                h_prev = np.zeros((N, self.hidden_size), dtype=np.float32)

        # 1. 词嵌入
        embed_W = self.layers[0].params[0]
        x_vec = embed_W[x_token:x_token+1]   # (1, D)

        # 2. 单步循环前向 (支持 Vanilla RNN 与 Gated GRU)
        Wx, Wh, b = self.rnn_layer.params
        if self.rnn_type == "gru":
            cell = GRUCell(Wx, Wh, b)
            h_new = cell.forward(x_vec, h_prev)
        else:
            a = np.dot(x_vec, Wx) + np.dot(h_prev, Wh) + b
            h_new = np.tanh(a)                   # (1, H)

        # 3. 投影到词表未归一化 Logits
        affine_W, affine_b = self.layers[2].params
        logits = np.dot(h_new, affine_W) + affine_b # (1, V)
        return logits.reshape(-1), h_new

    def predict_next_token_probs(self, x_token: int, h_prev: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        单步预测下一个 Token 的概率分布 (用于自回归生成或交互探针)

        参数:
            x_token: 单个词/字符的索引 (int)
            h_prev: 上一步的隐藏状态 (1, H), 若为 None 则使用内部状态或全 0
        返回:
            (probs, h_new): shape=(V,), shape=(1, H)
        """
        logits, h_new = self.predict_next_token_logits(x_token, h_prev)
        logits = logits - np.max(logits)
        exp_logits = np.exp(logits)
        probs = exp_logits / np.sum(exp_logits)
        return probs, h_new

    def generate(
        self,
        start_tokens: List[int],
        max_length: int = 50,
        temperature: float = 0.7,
        top_k: int = 0,
        top_p: float = 0.0,
        repetition_penalty: float = 1.0,
        repetition_window: int = 30
    ) -> List[int]:
        """
        自回归文本生成 (Autoregressive Generation)
        集成现代语言模型核心解码与采样策略：
        - 隐藏状态预热 (Warm-up Context): 严格保证因果时序对齐，杜绝末尾词重复喂入
        - 重复惩罚 (Repetition Penalty): 抑制语言模型陷入局部死循环与复读机退化
        - 温度调节 (Temperature): 控制采样概率平滑度 (<=0 为贪心解码，0.7 适中自然)
        - Top-K 截断采样: 保留概率最高的前 K 个候选
        - Top-P (Nucleus 核采样): 保留累积概率质量达到 P 的动态候选核

        参数:
            start_tokens: 提示引导词序列 (Prompt IDs)
            max_length: 生成最大长度
            temperature: 采样温度 (<=0 或接近 0 为贪心解码 argmax)
            top_k: Top-K 截断采样 (0 表示不截断)
            top_p: Top-P (Nucleus) 核采样阈值 (0.0 表示不启用)
            repetition_penalty: 重复惩罚系数 (1.0 表示不惩罚, >1.0 降低已出现词概率)
            repetition_window: 重复惩罚回溯窗口大小
        """
        if not start_tokens:
            raise ValueError("start_tokens 不能为空！")

        generated = list(start_tokens)
        h = None

        # 1. 预热隐藏状态：将 Prompt 中前 N-1 个词依次喂入网络接力
        if len(start_tokens) > 1:
            for token in start_tokens[:-1]:
                _, h = self.predict_next_token_probs(token, h)

        cur_token = start_tokens[-1]

        # 2. 逐步自回归生成后续 Token
        for _ in range(max_length):
            logits, h = self.predict_next_token_logits(cur_token, h)
            logits = logits.copy()

            # (A) 重复惩罚 (Repetition Penalty, CTRL 算法)
            if repetition_penalty != 1.0:
                recent_context = set(generated[-repetition_window:] if repetition_window > 0 else generated)
                for tok_id in recent_context:
                    if logits[tok_id] > 0:
                        logits[tok_id] /= repetition_penalty
                    else:
                        logits[tok_id] *= repetition_penalty

            # (B) 贪心解码 (若 temperature <= 1e-4)
            if temperature <= 1e-4:
                next_token = int(np.argmax(logits))
                generated.append(next_token)
                cur_token = next_token
                continue

            # (C) 温度调节
            logits = logits / max(temperature, 1e-4)
            logits_shifted = logits - np.max(logits)
            exp_logits = np.exp(logits_shifted)
            probs = exp_logits / np.sum(exp_logits)

            # (D) Top-K 截断过滤
            if 0 < top_k < len(probs):
                indices_to_remove = np.argsort(probs)[:-top_k]
                probs[indices_to_remove] = 0.0
                prob_sum = np.sum(probs)
                if prob_sum > 0:
                    probs = probs / prob_sum
                else:
                    probs = np.ones_like(probs) / len(probs)

            # (E) Top-P (Nucleus) 核采样
            if 0.0 < top_p < 1.0:
                sorted_indices = np.argsort(probs)[::-1]
                sorted_probs = probs[sorted_indices]
                cum_probs = np.cumsum(sorted_probs)

                cutoff = np.searchsorted(cum_probs, top_p)
                valid_indices = sorted_indices[:cutoff + 1]

                new_probs = np.zeros_like(probs)
                new_probs[valid_indices] = probs[valid_indices]
                prob_sum = np.sum(new_probs)
                if prob_sum > 0:
                    probs = new_probs / prob_sum
                else:
                    probs = np.ones_like(probs) / len(probs)

            # (F) 依概率采样
            next_token = int(np.random.choice(len(probs), p=probs))
            generated.append(next_token)
            cur_token = next_token

        return generated
