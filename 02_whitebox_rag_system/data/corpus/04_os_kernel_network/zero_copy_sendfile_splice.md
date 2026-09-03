# 零拷贝进阶：sendfile 与 splice 管道通道

## 1. sendfile() 系统调用
- 专门用于“文件到网络”的传输。
- 数据直接在内核层从文件 Page Cache 拷贝到 Socket 缓冲区，全程无需用户态参与，减少到 2 次上下文切换和 1 次 CPU 拷贝。
- **SG-DMA (Scatter-Gather DMA) 终极优化**：若硬件网卡支持 SG-DMA，内核仅需将 Page Cache 的物理内存地址和长度写入 Socket Buffer 描述符，网卡 DMA 直接从 Page Cache 读数据发往网络，**实现真正的零 CPU 拷贝 (True Zero-Copy)**。

## 2. splice() 系统调用
利用 Linux 内核内部的管道（Pipe）缓冲区移动数据指针，支持两个任意文件描述符之间的零拷贝数据定向传输。
