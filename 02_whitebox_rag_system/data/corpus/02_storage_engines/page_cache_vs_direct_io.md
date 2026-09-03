# 操作系统页缓存 (Page Cache) 与直接 I/O (O_DIRECT) 抉择

## 1. 双重缓存困境 (Double Buffering)
数据库为了实现精确的并发控制和事务隔离，在用户空间维护了庞大的 Buffer Pool（如 128GB）。若通过标准 OS 调用读写，数据将在 Linux Page Cache 和 Buffer Pool 中各存一份，浪费一倍内存空间。

## 2. 直接 I/O (Direct I/O) 的绕行机制
- 打开文件时指定 `O_DIRECT` 标志，使读取和写入直接在用户内存与磁盘驱动器之间传输，完全绕过 Linux Page Cache。
- 允许数据库开发人员自主控制预读策略、脏页替换与顺序合并，彻底消除内核态上下文切换和不必要的内存拷贝。
