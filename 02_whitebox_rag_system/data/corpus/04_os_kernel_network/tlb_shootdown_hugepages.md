# TLB 缓存失效风暴与大页内存 (HugePages) 优化

## 1. TLB (Translation Lookaside Buffer) 快表
CPU 内部专用的硬件缓存，缓存最近使用的虚拟页到物理页的映射关系，将访存开销从多次页表遍历减少到 1 次。

## 2. 多核并发下的 TLB Shootdown (击落风暴)
当某个核心修改了页表（如内存回收、页合并），为了保证缓存一致性，它必须向其他所有 CPU 核心发送跨处理器中断（Inter-Processor Interrupt, IPI），强制其他核心刷新各自的 TLB。在高并发多核服务器上，这种 IPI 中断会造成严重的 CPU 算力抖动。

## 3. 大页内存 (2MB / 1GB HugePages)
- 将单页大小从 4KB 扩大到 2MB 或 1GB。
- 页表级数减少，单条 TLB 能够覆盖的内存跨度提升 500 倍以上，极大幅度降低 TLB Miss 率，广泛应用于 DPDK、Redis 和大型数据库。
