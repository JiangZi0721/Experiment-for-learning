# 数据库压缩算法演进：Snappy 的低延迟 vs ZSTD 的极致压缩比

## 1. Snappy：吞吐优先
- Google 开发，基于 LZ77 演进。不进行熵编码（Huffman 编码），追求极高解压缩吞吐（单核解压可达 500MB/s）。
- 常用于 LSM-Tree 的高层（如 $L_0, L_1$）或对查询延迟极度敏感的实时系统。

## 2. ZSTD (Zstandard)：压缩比与吞吐的现代巅峰
- Facebook 开发，结合了快速前缀匹配与有限状态熵（Finite State Entropy, FSE）。
- 提供 1 到 22 级的多档调节。在相同解压缩速度下，压缩比通常比 Snappy 高出 30%~50%，是底层冷数据（Cold Data）归档压缩的工业事实标准。
