# 零拷贝架构：mmap 内存映射与页缓存交互

## 1. 传统 read() 系统的四次上下文切换与四次拷贝
1. 磁盘 $\to$ 内核 Page Cache (DMA 拷贝)
2. 内核 Page Cache $\to$ 用户空间 Buffer (CPU 拷贝)
3. 用户 Buffer $\to$ 内核 Socket Buffer (CPU 拷贝)
4. 内核 Socket Buffer $\to$ 网卡驱动协议栈 (DMA 拷贝)
伴随 4 次用户态与内核态的上下文切换。

## 2. mmap() 优化机制
`mmap()` 将内核 Page Cache 的物理页直接映射进用户进程的虚拟地址空间。
- 应用程序可以直接通过指针读写该文件内容，**彻底消除了第 2 步的 CPU 内存拷贝**。
- 但依然存在用户态向 Socket 写入时的 CPU 拷贝以及潜在的 SIGBUS 信号崩溃风险。
