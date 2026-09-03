# PagedAttention 与 vLLM 显存虚拟分页管理

## 1. 传统 KV Cache 的显存碎片痛点
由于自回归生成长度事先未知，传统系统必须为每个请求预先分配一段**物理连续**的最大长度显存空间。导致严重的内部碎片（Internal Fragmentation，预分配未使用）和外部碎片，显存有效利用率不足 40%。

## 2. PagedAttention 操作系统级分页思想
- 借鉴虚拟内存页表（Page Table）机制，将连续的虚拟 Token 映射到不连续的物理内存块（Physical Block，通常每个块存 16 个 Token）。
- **动态按需分配**：每生成 16 个 Token 才申请一个新的物理块。
- **写时复制 (Copy-On-Write)**：在并行采样（Parallel Sampling）和束搜索（Beam Search）中，多个序列共享同一段 Prompt 的物理 KV 块，分支时才按需复制，显存利用率提升至 96% 以上。
