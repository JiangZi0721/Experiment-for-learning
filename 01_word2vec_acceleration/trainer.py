"""
=============================================================================
模块名称: trainer.py
核心功能: 模型训练器 (Trainer) 与训练生命周期管理
包含功能:
    1. 动态 Mini-batch 洗牌与切分
    2. 梯度裁剪与参数优化更新
    3. 实时训练进度、损失、吞吐率 (samples/s) 与耗时监控
    4. 词向量权重持久化保存 (Save/Load)
=============================================================================
"""

import time
import os
import pickle
import numpy as np
from typing import Dict, List, Optional
from optimizer import clip_grads


class Trainer:
    """
    Word2Vec 模型专用训练器

    参数:
        model: 模型实例 (CBOWModel 或 SkipGramModel)
        optimizer: 优化器实例 (Adam 或 SGD)
    """

    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.loss_list = []
        self.epoch_loss_list = []

    def fit(
        self,
        contexts: np.ndarray,
        target: np.ndarray,
        max_epoch: int = 10,
        batch_size: int = 128,
        max_grad_norm: float = 5.0,
        log_interval: int = 20
    ) -> List[float]:
        """
        开始执行模型训练循环

        参数:
            contexts (np.ndarray): 上下文数据，形状为 (data_size, context_size)
            target (np.ndarray): 目标词数据，形状为 (data_size,)
            max_epoch (int): 最大训练轮数
            batch_size (int): 每个批次的样本数
            max_grad_norm (float): 梯度裁剪阈值
            log_interval (int): 打印训练日志的迭代步长

        返回:
            loss_list (List[float]): 记录的各步损失列表
        """
        data_size = len(contexts)
        max_iters = data_size // batch_size

        print("\n" + "=" * 70)
        print(f"[*] 开始 Word2Vec 训练 | 样本总量: {data_size:,} | 批次大小: {batch_size}")
        print(f"    训练轮数 (Epochs): {max_epoch} | 每轮迭代步数: {max_iters:,}")
        print("=" * 70)

        start_time = time.time()
        total_steps = 0

        for epoch in range(1, max_epoch + 1):
            epoch_start_time = time.time()
            epoch_total_loss = 0.0

            # 每一个 epoch 开始前随机打乱样本索引，避免数据顺序引入的偏差
            idx = np.random.permutation(data_size)
            x_shuffled = contexts[idx]
            t_shuffled = target[idx]

            for iters in range(max_iters):
                step_start = time.time()

                # 切分当前 Mini-batch
                batch_x = x_shuffled[iters * batch_size:(iters + 1) * batch_size]
                batch_t = t_shuffled[iters * batch_size:(iters + 1) * batch_size]

                # 1. 前向传播计算损失
                loss = self.model.forward(batch_x, batch_t)
                epoch_total_loss += loss
                self.loss_list.append(loss)
                total_steps += 1

                # 2. 反向传播计算梯度
                self.model.backward()

                # 3. 梯度裁剪 (防止梯度爆炸)
                if max_grad_norm is not None:
                    clip_grads(self.model.grads, max_norm=max_grad_norm)

                # 4. 优化器更新参数
                self.optimizer.update(self.model.params, self.model.grads)

                # 5. 打印阶段性训练进度与吞吐量
                if (iters + 1) % log_interval == 0 or (iters + 1) == max_iters:
                    elapsed = time.time() - start_time
                    step_time = time.time() - step_start
                    samples_per_sec = batch_size / max(step_time, 1e-6)
                    print(
                        f"| Epoch {epoch:2d}/{max_epoch:2d} "
                        f"| 进度 {iters + 1:4d}/{max_iters:4d} "
                        f"| 耗时: {elapsed:6.1f}s "
                        f"| Batch Loss: {loss:6.4f} "
                        f"| 速度: {samples_per_sec:6.0f} samples/s |"
                    )

            epoch_time = time.time() - epoch_start_time
            avg_epoch_loss = epoch_total_loss / max_iters
            self.epoch_loss_list.append(avg_epoch_loss)

            print("-" * 70)
            print(
                f"[OK] [Epoch {epoch} 结束] "
                f"平均损失: {avg_epoch_loss:.4f} | "
                f"本轮用时: {epoch_time:.2f}s"
            )
            print("-" * 70)

        total_elapsed = time.time() - start_time
        print(f"\n[DONE] 全部训练完成！总耗时: {total_elapsed:.2f} 秒\n")
        return self.loss_list

    def save_model(
        self,
        filepath: str,
        word_to_id: Dict[str, int],
        id_to_word: Dict[int, str]
    ) -> None:
        """
        保存词向量和模型权重至本地磁盘

        参数:
            filepath (str): 保存文件路径 (.pkl 或 .npz)
            word_to_id (dict): 词到 ID 字典
            id_to_word (dict): ID 到词字典
        """
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        save_data = {
            "W_in": self.model.W_in,
            "W_out": self.model.W_out,
            "word_to_id": word_to_id,
            "id_to_word": id_to_word
        }
        with open(filepath, "wb") as f:
            pickle.dump(save_data, f)
        print(f"[Trainer] 模型与词向量已成功保存至: {filepath}")

    @staticmethod
    def load_model(filepath: str):
        """
        从本地磁盘加载词向量模型及词典映射

        参数:
            filepath (str): 模型文件路径

        返回:
            save_data (dict): 包含 W_in, W_out, word_to_id, id_to_word
        """
        with open(filepath, "rb") as f:
            save_data = pickle.load(f)
        print(f"[Trainer] 成功从 {filepath} 加载模型权重！")
        return save_data
