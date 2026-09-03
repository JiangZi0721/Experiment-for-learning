# Linux 新一代异步 I/O：io_uring 核心架构与无系统调用原理

## 1. 传统 AIO 与 epoll 的局限
- Linux 原生 AIO 仅支持 `O_DIRECT`，且无法与网络 Socket 协同。
- epoll 只能监控 Socket 就绪，但对于磁盘文件 I/O 依然会阻塞线程。

## 2. io_uring 双环形无锁缓冲区 (SQ & CQ)
io_uring 在用户空间和内核空间之间共享两块内存环形队列（Ring Buffer）：
- **提交队列 (Submission Queue, SQ)**：用户将 I/O 请求封装为 SQE (SQ Entry) 写入队列尾部。
- **完成队列 (Completion Queue, CQ)**：内核处理完成后将结果写入 CQE (CQ Entry)，用户直接从头部读取。

## 3. IORING_SETUP_SQPOLL 无系统调用模式
开启内核轮询线程（SQ Poll Thread）。用户写入 SQE 后无需执行任何系统调用（System Call），内核轮询线程自动抓取执行，**彻底消除上下文切换开销**，IOPS 相比传统方案翻倍。
