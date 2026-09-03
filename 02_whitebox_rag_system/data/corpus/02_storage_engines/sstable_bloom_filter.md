# SSTable 物理存储格式与布隆过滤器加速

## 1. SSTable 二进制物理布局
SSTable 通常划分为固定大小的数据块（Data Block，通常 4KB~64KB）：
- **Data Blocks**：按键有序存储的键值对，通常采用前缀压缩（Prefix Encoding）。
- **Index Block**：记录每个 Data Block 的最大键与文件偏移量，允许在内存中进行二分查找。
- **Filter Block**：包含整个 SSTable 的布隆过滤器（Bloom Filter）位图。
- **Footer**：位于文件末尾，记录 Index Block 和 Meta Index Block 的偏移量与 Magic Number。

## 2. 布隆过滤器的误报率与剪枝
在读取未命中某 SSTable 时，布隆过滤器能在 $O(1)$ 时间内确定“该 Key 绝对不存在”，从而跳过读取该文件磁盘 I/O。
- 公式：最佳位图大小 $m = -\frac{n \ln p}{(\ln 2)^2}$，通常每个 Key 分配 10 bit 时误报率约为 1%。
