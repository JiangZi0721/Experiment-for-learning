# CUDA Graph 静态图加速：消除小算子内核启动开销

## 1. CPU 提交瓶颈 (Kernel Launch Overhead)
大模型 Decode 阶段每生成一个 Token 需要串行调用数百个细粒度算子（LayerNorm、Add、RMSNorm 等）。每次 GPU 内核启动需要约 3~5 微秒的 CPU 调度开销。当批量较小时，CPU 提交速度甚至赶不上 GPU 执行速度。

## 2. CUDA Graph 预捕获机制
- 将这数百个内核调用及其内存依赖拓扑在预热阶段一次性捕获为一张静态的有向无环图（Graph）。
- 运行时，CPU 只需向 GPU 提交一个单一的工作单元执行该 Graph，内核调度开销归零，显著缩短单 Token 生成的端到端延迟。
