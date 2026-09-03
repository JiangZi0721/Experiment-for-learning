# DeepSpeed ZeRO 显存优化：从 ZeRO-1 到 ZeRO-3 全景拆解

## 1. 模型训练显存构成
显存主要被模型状态（Model States）占据：参数（Parameters $P$）、梯度（Gradients $G$）、优化器状态（Optimizer States $O$，Adam 中包含一阶动量和二阶动量，占用 $12\times P$ 显存）。

## 2. ZeRO 三阶段切片
- **ZeRO-1 (Optimizer State Partitioning, $P_{os}$)**：将 Adam 优化器状态均匀切分到 $N$ 个 GPU 上。显存减少 4 倍，通信开销为 0。
- **ZeRO-2 (Gradient Partitioning, $P_{os+g}$)**：进一步将梯度也在 GPU 间切分。每个 GPU 仅保留自身负责参数的梯度。显存减少 8 倍，通信开销无额外增加。
- **ZeRO-3 (Parameter Partitioning, $P_{os+g+p}$)**：**将模型参数本身也彻底打碎分区**。在前向传播计算某一层时，通过 `All-Gather` 动态拉取参数，算完立即释放内存；反向传播时再次拉取，算完再释放。允许单张消费级显卡训练百亿参数模型。
