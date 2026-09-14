# 实验三：经典基础 RNN (Vanilla RNN) 下的 Seq2Seq 优化对比

本实验以最纯粹的 **经典基础循环神经网络（Elman / Vanilla RNN: $h_t = \tanh(W x_t + U h_{t-1})$）** 为底层架构，完全去除了 LSTM 的门控机制与 Cell 状态累加通道，以探究 **Reverse** 与 **Peeky** 在极端梯度衰减条件下的表现。

---

## 目录结构说明

```text
03_Vanilla_RNN_Addition/
├── dataset.py                  # 加法数据生成与加载器 (0~999 加法)
├── models.py                   # 纯 Vanilla RNN 实现 (RNNEncoder, RNNStandardDecoder, RNNPeekyDecoder)
├── train.py                    # 4 组 Vanilla RNN 模型完整训练与 Exact Match 评测脚本
├── visualize.py                # 绘制 Vanilla RNN 学习曲线与汇总脚本
├── rnn_results.json            # 4 组模型 25 轮完整原始日志数据
├── summary_report.md           # 实验评估表格与典型算式错例对比报告
├── rnn_accuracy_comparison.png # 验证集 Exact Match 准确率与 Loss 曲线对比图
└── README.md                   # 实验专属说明与运行指南
```

---

## 实验结果对比 (25 轮严格实测)

| 组别 | 模型架构 | 输入顺序 | 解码器 | 参数量 | 训练耗时 (s) | 最终验证 Loss | **Exact Match 全对率** |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **RNN_Baseline** | Vanilla RNN | 正向 | 标准解码器 | 39,469 | 48.2s | 0.4623 | **30.18%** |
| **RNN_Reverse** | Vanilla RNN | 逆序 | 标准解码器 | 39,469 | 46.6s | **0.0702** | **92.72%** (+62.54%) |
| **RNN_Peeky** | Vanilla RNN | 正向 | Peeky 解码器 | 57,517 | 36.9s | 0.0928 | **87.82%** (+57.64%) |
| **RNN_Reverse+Peeky** | Vanilla RNN | 逆序 | Peeky 解码器 | 57,517 | 39.4s | **0.0485** | **94.10%** (+63.92%) |

---

## 核心理论发现：揭开教材中“Reverse 神奇现象”的真正谜底！

在经典教材（如《深度学习进阶：自然语言处理》第 7 章）中，作者展示了“加法任务中引入 Reverse 后准确率瞬间暴增”的经典图景。而在我们本轮 Vanilla RNN 实测中，这一神迹被**100% 完美复现**：

1. **为什么在基础 RNN 中，Reverse 能够从 30% 暴涨至 92.7%？**
   - 基础 RNN 没有 LSTM 的线性累加状态通道 $c_t$，反向传播时每一时间步都面临雅可比矩阵的连乘（$\lambda^T$ 梯度指数消失）。
   - 在正向输入时，第 1 个字符的梯度要跨越 11 个时间步，梯度完全衰减为 0，模型无法更新最初的几位！
   - 而 **Reverse 将输入序列逆序后，首端字符与解码起点的距离被缩短为 1！**
   - 极短的传播路径让脆弱的 Vanilla RNN 瞬间打破了梯度消失的封锁，仅用 **5 轮** 就突破了 **83.8%** 的准确率！

2. **为什么现代 PyTorch LSTM 没有这么夸张？**
   - PyTorch 的标准 LSTM 完整传递了 Cell 状态 $c_n$（长程梯度高速公路），即便不反转也具有极强记忆，因此 LSTM 的瓶颈在于**跨步上下文的直接读取（Peeky）**；
   - 而基础 RNN（以及教材中丢弃了 $c$ 状态的手写简陋 LSTM）没有 Cell 状态通道，**Reverse 对它们而言是决定生死存亡的关键阶跃**！
