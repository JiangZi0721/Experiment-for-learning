# LSM-Tree 核心架构与三大放大效应

## 1. LSM-Tree 存储分层与写入流水线
LSM-Tree (Log-Structured Merge-tree) 采用“追加写”换取高吞吐写入性能：
1. **内存 MemTable**：写入首先记录 WAL 保证持久性，然后插入内存跳表（Skiplist）。
2. **只读 Immutable MemTable**：MemTable 达到阈值后冻结为只读，后台异步线程执行 Flush。
3. **分层磁盘 SSTable**：Flush 落盘形成有序字符串表（Sorted String Table, SSTable）。

## 2. 三大放大效应 (Trade-offs)
- **写放大 (Write Amplification, WAF)**：实际写入物理磁盘的字节数与用户逻辑写入字节数的比值。Compaction 过程反复读取并重写数据，WAF 通常高达 10~30。
- **读放大 (Read Amplification, RAF)**：一次点查可能需要遍历 MemTable、Bloom Filter 以及多个层级（Levels）的 SSTable，导致数十次磁盘 I/O。
- **空间放大 (Space Amplification, SAF)**：由于数据覆写与删除采用追加墓碑标记（Tombstone），旧版本与无效数据在 Compaction 前占用额外磁盘空间。
