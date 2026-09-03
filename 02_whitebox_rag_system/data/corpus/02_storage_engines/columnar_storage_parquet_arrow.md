# 列式存储物理格局：Apache Parquet 与 Apache Arrow 对比

## 1. 行存 (OLTP) vs 列存 (OLAP)
- **行存**：同一行的数据物理连续，适合单行 CRUD、点查。但 OLAP 分析聚合查询时会产生海量无效列的磁盘 I/O。
- **列存**：同一列的数据连续存储，聚合 `SUM(salary)` 只需读取一列数据，磁盘扫描量骤减 90%。

## 2. Parquet (磁盘级存储)
- 基于 Dremel 嵌套结构的二进制文件格式。按 Row Group、Column Chunk 和 Page 分层，支持字典编码、Run-Length 编码（RLE）与 Snappy/ZSTD 极高压缩率。

## 3. Arrow (内存中向量化执行)
- 跨语言的标准内存列式布局。通过对齐的扁平缓冲区（Contiguous Buffer）实现零反序列化拷贝，配合 CPU SIMD 向量化指令实现极致计算加速。
