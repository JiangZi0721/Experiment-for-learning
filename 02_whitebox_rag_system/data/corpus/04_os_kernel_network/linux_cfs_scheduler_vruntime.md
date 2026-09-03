# Linux CFS 完全公平调度器与虚拟运行时间 (vruntime)

## 1. 完全公平的核心哲学
CFS 放弃了传统的固定时间片分配机制，采用红黑树动态维护就绪队列中所有任务的虚拟运行时间（`vruntime`）。

## 2. 虚拟运行时间换算公式
$$\Delta vruntime = \Delta exec\_time \times \frac{\text{NICE\_0\_LOAD}}{\text{se}->load.weight}$$
- 权重越高的进程（Nice 值越小，优先级越高），其物理运行时间换算出的 $vruntime$ 增长得越慢。
- CFS 调度器在每次时钟中断或调度时，**永远优先选择红黑树最左侧（$vruntime$ 最小）的节点执行**，从而在宏观上达到极度精确的时间片加权分配。
