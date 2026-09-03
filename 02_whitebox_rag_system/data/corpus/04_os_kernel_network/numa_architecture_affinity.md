# NUMA 架构内存不均匀访问与 CPU 亲和性 (Affinity) 绑定

## 1. SMP 对称多处理 vs NUMA 非一致性内存访问
- **SMP (Symmetric Multi-Processing)**：所有 CPU 共享同一条内存总线，随着 CPU 核心数增多总线成为严重瓶颈。
- **NUMA**：将 CPU 与内存划分为多个 Node。CPU 访问本地 Node 内存延迟极低；访问远端 Node 内存需要经过跨芯片互联总线（如 QPI/UPI），延迟高出 2~3 倍。

## 2. CPU 亲和性 (CPU Affinity) 与隔离 (numactl)
在高并发低延迟中间件中，若线程被操作系统调度器频繁调度到跨 Node 的核心上执行，会导致频繁的跨 Node 访存和缓存失效率。
- 使用 `pthread_setaffinity_np` 绑定工作线程到特定 CPU 核。
- 使用 `numactl --interleave` 或 `--cpunodebind` 指定内存分配策略。
