# 大模型量化前沿：AWQ 激活感知量化 vs GPTQ 二阶误差优化

## 1. 权重离散化的挑战
将 FP16 权重截断为 INT4 时，极易导致大模型在特定任务上困惑度（Perplexity）剧增甚至胡言乱语。

## 2. GPTQ (基于二阶海森矩阵优化)
- 借鉴 Optimal Brain Surgeon (OBS) 思想，利用逆海森矩阵（Hessian）计算量化某一行权重对全局输出造成的误差，并动态补偿调整尚未量化的相邻权重。

## 3. AWQ (Activation-aware Weight Quantization)
- 核心洞察：**权重并不等权重要，受大激活值（Activation Outliers）影响的仅占 1% 的显著权重才是维持模型精度的命脉**。
- 策略：保护这 1% 的显著权重不被过度量化，或者在量化前根据激活值分布对权重矩阵执行全局平滑缩放，推理开销极低。
