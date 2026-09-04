# -*- coding: utf-8 -*-
"""
Truncated BPTT 训练器与时序对齐批次加载器
严格实现时间连续切分 (Time-Aligned Sequential Chunking)，实现前向状态跨块接力与反向梯度边界截断。
"""
from typing import List, Dict, Any, Tuple, Optional, Callable
import numpy as np
from src.rnn_lm import RNNLM
from src.layers import clip_grads


class ContinuousCorpusLoader:
    """
    时序连续语料切分加载器 (Batch-Level Sequential Aligned Loader)

    原理剖析:
        为了在 Truncated BPTT 中使隐藏状态能够跨 Chunk 有效延续，
        必须将语料平均切分为 N 份 (N = batch_size)，每个样本流在各自的轨道上串行向后推进。
        这样在步推进过程中，batch[i] 上的数据恰好是上一步 batch[i] 紧随其后的上下文！
    """
    def __init__(self, corpus: np.ndarray, batch_size: int, time_size: int):
        self.corpus = corpus
        self.batch_size = batch_size
        self.time_size = time_size

        self.data_size = len(corpus)
        # 计算每个 batch 样本分配到的序列长度
        self.jump = self.data_size // batch_size
        # 计算总共可以切分出多少个时序块 (Chunks)
        self.max_iters = (self.jump - 1) // time_size

        # 初始化每个 batch 样本在语料中的起始指针
        self.offsets = [i * self.jump for i in range(batch_size)]
        self.current_iter = 0

    def __iter__(self):
        self.current_iter = 0
        return self

    def __next__(self) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        提取当前步的时序块 (Chunk)
        返回:
            xs: 输入序列矩阵 (N, T)
            ts: 目标预测标签矩阵 (N, T) (右移 1 位)
            meta: 元数据探针 (当前批次时序位置)
        """
        if self.current_iter >= self.max_iters:
            raise StopIteration

        N, T = self.batch_size, self.time_size
        xs = np.empty((N, T), dtype=np.int32)
        ts = np.empty((N, T), dtype=np.int32)

        time_offsets = []
        for i, offset in enumerate(self.offsets):
            start = offset + self.current_iter * T
            end = start + T
            xs[i] = self.corpus[start:end]
            ts[i] = self.corpus[start+1:end+1]
            time_offsets.append((start, end))

        meta = {
            "chunk_idx": self.current_iter,
            "max_chunks": self.max_iters,
            "time_offsets": time_offsets,
        }

        self.current_iter += 1
        return xs, ts, meta


class SGD:
    """带学习率调度的随机梯度下降"""
    def __init__(self, lr: float = 0.1):
        self.lr = lr

    def update(self, params: List[np.ndarray], grads: List[np.ndarray]):
        for p, g in zip(params, grads):
            p -= self.lr * g


class Adam:
    """白盒纯 NumPy 实现的自适应矩估计优化器 (Adam)"""
    def __init__(self, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None

    def update(self, params: List[np.ndarray], grads: List[np.ndarray]):
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]

        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2 ** self.iter) / (1.0 - self.beta1 ** self.iter)

        for i in range(len(params)):
            self.m[i] += (1 - self.beta1) * (grads[i] - self.m[i])
            self.v[i] += (1 - self.beta2) * (grads[i] ** 2 - self.v[i])
            params[i] -= lr_t * self.m[i] / (np.sqrt(self.v[i]) + 1e-7)


class RnnlmTrainer:
    """
    Truncated BPTT 教学级白盒训练器
    全流程监控每个 Chunk 的前向延续、反向截断、梯度范数与困惑度蜕变
    """
    def __init__(
        self,
        model: RNNLM,
        optimizer: Any,
        max_grad_norm: float = 5.0,
        probe_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ):
        self.model = model
        self.optimizer = optimizer
        self.max_grad_norm = max_grad_norm
        self.probe_callback = probe_callback

        self.loss_history: List[float] = []
        self.ppl_history: List[float] = []
        self.grad_norm_history: List[float] = []

    def fit(
        self,
        corpus: np.ndarray,
        batch_size: int = 16,
        time_size: int = 20,
        max_epoch: int = 10,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        启动 Truncated BPTT 迭代训练
        """
        loader = ContinuousCorpusLoader(corpus, batch_size, time_size)
        total_steps = 0

        for epoch in range(max_epoch):
            # 每个 Epoch 开始时重置模型内部的时序状态
            self.model.reset_state()
            epoch_loss = 0.0
            chunk_count = 0

            for xs, ts, meta in loader:
                # 1. 记录前向跨块接力前状态 (Chunk Relay Probe)
                prev_state = self.model.get_state()
                prev_state_norm = float(np.linalg.norm(prev_state)) if prev_state is not None else 0.0

                # 2. 前向传播 (在 TimeRNN 内部 stateful=True 自动继承 prev_state)
                loss = self.model.forward(xs, ts)

                # 3. 记录前向更新后的状态
                curr_state = self.model.get_state()
                curr_state_norm = float(np.linalg.norm(curr_state)) if curr_state is not None else 0.0

                # 4. 反向传播 (BPTT 逆流)
                self.model.backward()

                # 获取在 Chunk 边界处向更早历史回传的梯度 (Truncated 截断点)
                # 这个梯度在标准 Truncated BPTT 中被果断舍弃，不向前一个 Chunk 回传！
                dh_to_prev_chunk = self.model.rnn_layer.dh
                dh_truncated_norm = float(np.linalg.norm(dh_to_prev_chunk)) if dh_to_prev_chunk is not None else 0.0

                # 5. 梯度裁剪 (防止梯度爆炸)
                orig_gnorm, final_gnorm, scale_rate = clip_grads(self.model.grads, self.max_grad_norm)

                # 6. 优化器参数更新
                self.optimizer.update(self.model.params, self.model.grads)

                # 7. 指标累计
                ppl = float(np.exp(loss)) if loss < 50 else float("inf")
                epoch_loss += loss
                chunk_count += 1
                total_steps += 1

                self.loss_history.append(loss)
                self.ppl_history.append(ppl)
                self.grad_norm_history.append(orig_gnorm)

                # 8. 探针数据回调
                probe_payload = {
                    "epoch": epoch + 1,
                    "total_epochs": max_epoch,
                    "step": total_steps,
                    "chunk_idx": meta["chunk_idx"],
                    "max_chunks": meta["max_chunks"],
                    "loss": loss,
                    "ppl": ppl,
                    "orig_grad_norm": orig_gnorm,
                    "final_grad_norm": final_gnorm,
                    "grad_scale_rate": scale_rate,
                    "prev_h_norm": prev_state_norm,
                    "curr_h_norm": curr_state_norm,
                    "dh_truncated_norm": dh_truncated_norm,
                    "time_rnn_probe": self.model.rnn_layer.probe_data,
                    "loss_probe": self.model.loss_layer.probe_data,
                }

                if self.probe_callback is not None:
                    self.probe_callback(probe_payload)

            avg_loss = epoch_loss / max(1, chunk_count)
            avg_ppl = float(np.exp(avg_loss)) if avg_loss < 50 else float("inf")

        return {
            "loss_history": self.loss_history,
            "ppl_history": self.ppl_history,
            "grad_norm_history": self.grad_norm_history,
        }
