# 流水线并行 (Pipeline Parallelism): GPipe 气泡与 1F1B 调度

## 1. 流水线气泡 (Bubble) 痛点
将深度为 $L$ 的网络按层划分为多个阶段（Stages）分配给不同的 GPU。初级朴素流水线中，后级 GPU 必须等待前级输出，导致大量计算单元空闲等待（Bubble）。

## 2. GPipe 与微批次 (Micro-batch)
GPipe 将全局 Batch 切分为 $M$ 个 Micro-batch，使各个阶段能够交叠计算，将气泡比例压缩至 $\frac{K-1}{M+K-1}$。

## 3. 1F1B 稳态调度 (One Forward, One Backward)
在稳态运行期，每张 GPU 交替执行一次前向计算（Forward）和一次反向计算（Backward）。这使得反向传播尽早释放前向激活值内存，峰值显存占用大幅低于 GPipe。
