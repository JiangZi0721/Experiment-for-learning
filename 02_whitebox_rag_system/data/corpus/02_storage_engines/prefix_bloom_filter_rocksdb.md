# RocksDB 前缀布隆过滤器 (Prefix Bloom Filter) 优化

## 1. 传统全量布隆过滤器的局限
全量 Bloom Filter 只能针对完整 Key 进行存在性判定，对范围扫描（如 `iterator.Seek("user_1001_order_*")`）完全无效，每次范围扫描必须进入磁盘。

## 2. 前缀哈希与范围跳过
- 用户自定义前缀切分函数（如截取前 12 个字节作为前缀）。
- RocksDB 在构建 SSTable 时，针对 Key 的前缀单独计算哈希并放入 Prefix Bloom Filter。
- 执行前缀 Seek 操作时，若该前缀在 Bloom Filter 中判定不存在，直接跳过整个 SSTable，极大优化时序与多维查询。
