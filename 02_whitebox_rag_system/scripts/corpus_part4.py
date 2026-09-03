# -*- coding: utf-8 -*-
"""04_os_kernel_network 25 topics"""

OS_KERNEL_NETWORK = [
    ("linux_virtual_memory_page_tables", "Linux 虚拟内存系统与四级页表映射", """# Linux 虚拟内存系统与四级页表映射

## 1. 虚拟地址空间与 MMU
- 现代 64 位操作系统（如 x86_64）使用 48 位虚拟地址，划分为用户空间（User Space, $0\\sim 0x00007FFFFFFFFFFF$）和内核空间（Kernel Space, $0xFFFF800000000000\\sim 0xFFFFFFFFFFFFFFFF$）。
- 内存管理单元（MMU）负责将虚拟地址（VA）翻译为物理内存地址（PA）。

## 2. 四级页表结构 (PGD, P4D, PUD, PMD, PTE)
为了避免单级线性页表占用几百 GB 连续内存，Linux 采用多级稀疏树状页表：
1. **PGD (Page Global Directory)**：顶层全局页目录，CR3 寄存器存放其基地址。
2. **P4D (Page 4th Directory)**：支持 5 级分页的预留层。
3. **PUD (Page Upper Directory)**：页上级目录。
4. **PMD (Page Middle Directory)**：页中间目录。
5. **PTE (Page Table Entry)**：页表项，映射到真正的 4KB 物理页框（Page Frame），包含 Present, R/W, User/Supervisor 等控制标志。"""),

    ("tlb_shootdown_hugepages", "TLB 缓存失效风暴与大页内存 (HugePages) 优化", """# TLB 缓存失效风暴与大页内存 (HugePages) 优化

## 1. TLB (Translation Lookaside Buffer) 快表
CPU 内部专用的硬件缓存，缓存最近使用的虚拟页到物理页的映射关系，将访存开销从多次页表遍历减少到 1 次。

## 2. 多核并发下的 TLB Shootdown (击落风暴)
当某个核心修改了页表（如内存回收、页合并），为了保证缓存一致性，它必须向其他所有 CPU 核心发送跨处理器中断（Inter-Processor Interrupt, IPI），强制其他核心刷新各自的 TLB。在高并发多核服务器上，这种 IPI 中断会造成严重的 CPU 算力抖动。

## 3. 大页内存 (2MB / 1GB HugePages)
- 将单页大小从 4KB 扩大到 2MB 或 1GB。
- 页表级数减少，单条 TLB 能够覆盖的内存跨度提升 500 倍以上，极大幅度降低 TLB Miss 率，广泛应用于 DPDK、Redis 和大型数据库。"""),

    ("page_fault_handling_anonymous_file", "Linux 缺页异常处理机制：匿名页 vs 文件页", """# Linux 缺页异常处理机制：匿名页 vs 文件页

## 1. 延时分配机制 (Lazy Allocation)
当用户调用 `malloc()` 或 `mmap()` 时，内核仅仅在进程的 `vm_area_struct` 中记录一段虚地址范围，并不实际分配物理内存。只有当进程第一次对该地址进行读写时，CPU 触发 14 号异常——缺页中断（Page Fault）。

## 2. 两大类缺页路径
- **匿名页缺页 (Anonymous Page Fault)**：如堆（Heap）、栈（Stack）和私有写拷贝内存。内核直接从物理空闲伙伴系统（Buddy System）分配一个清零页建立映射。
- **文件页缺页 (File-backed Page Fault)**：如可执行代码段、通过 `mmap` 映射的文件。内核必须先在 Page Cache 中查找；若未命中，发起磁盘驱动器 I/O 将文件扇区加载进页缓存，再建立 PTE 映射。"""),

    ("zero_copy_mmap_pagecache", "零拷贝架构：mmap 内存映射与页缓存交互", """# 零拷贝架构：mmap 内存映射与页缓存交互

## 1. 传统 read() 系统的四次上下文切换与四次拷贝
1. 磁盘 $\\to$ 内核 Page Cache (DMA 拷贝)
2. 内核 Page Cache $\\to$ 用户空间 Buffer (CPU 拷贝)
3. 用户 Buffer $\\to$ 内核 Socket Buffer (CPU 拷贝)
4. 内核 Socket Buffer $\\to$ 网卡驱动协议栈 (DMA 拷贝)
伴随 4 次用户态与内核态的上下文切换。

## 2. mmap() 优化机制
`mmap()` 将内核 Page Cache 的物理页直接映射进用户进程的虚拟地址空间。
- 应用程序可以直接通过指针读写该文件内容，**彻底消除了第 2 步的 CPU 内存拷贝**。
- 但依然存在用户态向 Socket 写入时的 CPU 拷贝以及潜在的 SIGBUS 信号崩溃风险。"""),

    ("zero_copy_sendfile_splice", "零拷贝进阶：sendfile 与 splice 管道通道", """# 零拷贝进阶：sendfile 与 splice 管道通道

## 1. sendfile() 系统调用
- 专门用于“文件到网络”的传输。
- 数据直接在内核层从文件 Page Cache 拷贝到 Socket 缓冲区，全程无需用户态参与，减少到 2 次上下文切换和 1 次 CPU 拷贝。
- **SG-DMA (Scatter-Gather DMA) 终极优化**：若硬件网卡支持 SG-DMA，内核仅需将 Page Cache 的物理内存地址和长度写入 Socket Buffer 描述符，网卡 DMA 直接从 Page Cache 读数据发往网络，**实现真正的零 CPU 拷贝 (True Zero-Copy)**。

## 2. splice() 系统调用
利用 Linux 内核内部的管道（Pipe）缓冲区移动数据指针，支持两个任意文件描述符之间的零拷贝数据定向传输。"""),

    ("epoll_lt_vs_et_triggering", "I/O 多路复用深入：epoll 水平触发 (LT) vs 边缘触发 (ET)", """# I/O 多路复用深入：epoll 水平触发 (LT) vs 边缘触发 (ET)

## 1. Level-Triggered (水平触发, LT - 默认模式)
- 只要文件描述符的读缓冲区中有未读尽的数据，或者写缓冲区有空闲，每次调用 `epoll_wait()` 都会持续反复通知就绪。
- **优点**：不易丢事件，容错率高。
- **缺点**：若数据一次没读完，频繁唤醒产生不必要的内核轮询开销。

## 2. Edge-Triggered (边缘触发, ET - 高性能模式)
- 只有在状态发生变化（从未就绪到就绪，或有新数据到达）的瞬间，`epoll_wait()` 才会通知一次。
- **开发铁律**：
  1. 相关的 socket 必须设置为非阻塞模式（`O_NONBLOCK`）。
  2. 接收数据时，必须在循环中执行 `read()`，**直到返回 `EAGAIN` 或 `EWOULDBLOCK` 为止**。否则遗留在缓冲区中的数据将永远无法触发下一次唤醒，造成严重请求挂死。"""),

    ("io_uring_sqe_cqe_architecture", "Linux 新一代异步 I/O：io_uring 核心架构与无系统调用原理", """# Linux 新一代异步 I/O：io_uring 核心架构与无系统调用原理

## 1. 传统 AIO 与 epoll 的局限
- Linux 原生 AIO 仅支持 `O_DIRECT`，且无法与网络 Socket 协同。
- epoll 只能监控 Socket 就绪，但对于磁盘文件 I/O 依然会阻塞线程。

## 2. io_uring 双环形无锁缓冲区 (SQ & CQ)
io_uring 在用户空间和内核空间之间共享两块内存环形队列（Ring Buffer）：
- **提交队列 (Submission Queue, SQ)**：用户将 I/O 请求封装为 SQE (SQ Entry) 写入队列尾部。
- **完成队列 (Completion Queue, CQ)**：内核处理完成后将结果写入 CQE (CQ Entry)，用户直接从头部读取。

## 3. IORING_SETUP_SQPOLL 无系统调用模式
开启内核轮询线程（SQ Poll Thread）。用户写入 SQE 后无需执行任何系统调用（System Call），内核轮询线程自动抓取执行，**彻底消除上下文切换开销**，IOPS 相比传统方案翻倍。"""),

    ("tcp_three_way_handshake_syn_flood", "TCP 三次握手状态迁移与 SYN Flood 洪水攻击防御", """# TCP 三次握手状态迁移与 SYN Flood 洪水攻击防御

## 1. 三次握手状态机
1. Client 发送 `SYN` (seq=x)，进入 `SYN_SENT`。
2. Server 回复 `SYN+ACK` (seq=y, ack=x+1)，进入 `SYN_RCVD`，将连接放入**半连接队列 (SYN Queue)**。
3. Client 回复 `ACK` (seq=x+1, ack=y+1)，进入 `ESTABLISHED`；Server 收到后将连接移入**全连接队列 (Accept Queue)**。

## 2. SYN Flood 攻击本质
黑客伪造大量虚假源 IP 发送大量 SYN 请求，且故意不回复最后的 ACK。导致 Server 的半连接队列迅速打满，正常合法用户的连接请求全部被丢弃。

## 3. SYN Cookie 防御机制
- 当半连接队列满时，Server 不分配 `struct request_sock` 结构。
- 依据时间戳、源/目的 IP、端口以及安全密钥通过哈希计算出初始序列号 $seq_y$（即 SYN Cookie）。
- 只有当合法的第三次 ACK 到达且其中的 $ack$ 能被成功还原验证时，才分配连接资源，彻底免受队列占满限制。"""),

    ("tcp_time_wait_reuseaddr", "TCP 四次挥手 TIME_WAIT 状态机与 SO_REUSEADDR 机制", """# TCP 四次挥手 TIME_WAIT 状态机与 SO_REUSEADDR 机制

## 1. 为何必须维持 2MSL 的 TIME_WAIT 状态
主动关闭连接的一方在收到对端 FIN 并回复 ACK 后，必须在此状态等待 $2\\times \\text{MSL}$（最大报文生存时间，Linux 默认 60s）：
- **可靠终止连接**：防止最后发出的 ACK 报文丢失，确保如果对端因超时重发 FIN，本端仍能回复 ACK。
- **消除迷走报文干扰**：保证本连接在网络中延迟残留的所有历史旧报文全部自然消亡，不至于污染后续复用相同四元组的新连接。

## 2. SO_REUSEADDR 与端口耗尽
高并发短连接服务器下，大量 Socket 卡在 TIME_WAIT 会耗尽可用端口。通过设置 `SO_REUSEADDR`，允许处于 TIME_WAIT 状态的端口被新创建的 Socket 立即绑定复用。"""),

    ("tcp_cubic_vs_bbr_congestion", "TCP 拥塞控制演进：基于丢包的 CUBIC vs 基于吞吐瓶颈的 BBR", """# TCP 拥塞控制演进：基于丢包的 CUBIC vs 基于吞吐瓶颈的 BBR

## 1. 传统 CUBIC (基于丢包的被动控制)
- 认为“丢包即拥塞”。将拥塞窗口（cwnd）按照三次函数增长，直到发生网络丢包才急剧减半窗口。
- **缓冲区膨胀 (Bufferbloat) 死穴**：在当今中间路由器拥有海量缓冲队列的环境下，CUBIC 会填满队列才丢包，导致网络往返时延（RTT）急剧恶化。

## 2. Google BBR (Bottleneck Bandwidth and RTT)
- 抛弃丢包信号，采用主动物理建模。
- **实时探测两个物理极值**：网络最大可用带宽（Max BtlBw）与最小传播往返时间（Min RTprop）。
- 将在网数据包量（In-Flight Data）严格控制在 $BDP = \\text{BtlBw} \\times \\text{RTprop}$，既打满链路带宽又绝不填塞路由器队列，在弱网或跨洋长延迟链路中吞吐量可提升数倍至数十倍。"""),

    ("socket_buffer_tuning_scale", "Socket 缓冲区内核调优与 TCP 窗口缩放因子 (Window Scale)", """# Socket 缓冲区内核调优与 TCP 窗口缩放因子 (Window Scale)

## 1. 带宽时延积 (BDP) 与窗口瓶颈
$$BDP = \\text{Bandwidth} \\times \\text{RTT}$$
为了使网络管道始终填满数据，发送端与接收端的滑动窗口大小必须至少等于 BDP。

## 2. 传统 16-bit 窗口限制与 Window Scale 扩展
TCP 报文头中原始的 `Window Size` 字段仅有 16 位，最大只能表示 64KB，在千兆万兆网络中极速变成瓶颈。
- **Window Scale (RFC 1323)**：在握手 SYN 阶段协商缩放移位因子（0~14）。实际窗口大小为 $\\text{Window} \\times 2^{scale}$，最大可扩展到 1GB。

## 3. 内核自动调谐 (Autotuning)
Linux 针对 `rmem` 和 `wmem` 提供了 `[min, default, max]` 三元组，内核根据当前连接的实时 RTT 和传输速率自动弹性调整 Socket 缓冲区。"""),

    ("ebpf_xdp_packet_processing", "eBPF 内核可编程性与 XDP 极速报文处理", """# eBPF 内核可编程性与 XDP 极速报文处理

## 1. 传统网络栈协议过滤的开销
数据包经过网卡驱动后，内核需要分配庞大的 `sk_buff` 内存结构体，经过复杂的网络层、路由表、Netfilter 防火墙后才到达应用层。

## 2. XDP (eXpress Data Path) 极致拦截
- 在网卡驱动层刚完成 DMA 填充、**尚未分配 `sk_buff` 的物理最前端**直接运行经安全沙盒验证的 eBPF 字节码。
- **直接动作**：`XDP_DROP`（单核线速丢弃每秒数千万包抗 DDoS）、`XDP_TX`（原路极速反弹转发）、`XDP_PASS`（送入上层网络栈）。性能比传统 iptables 提升一个数量级。"""),

    ("cpu_cache_lines_false_sharing", "CPU 缓存行对齐 (Cache Line) 与伪共享 (False Sharing) 陷阱", """# CPU 缓存行对齐 (Cache Line) 与伪共享 (False Sharing) 陷阱

## 1. 缓存行基本物理单位
现代 CPU（如 x86, ARM）的 L1/L2/L3 Cache 读写数据以 **64 字节** 的 Cache Line 为最小原子单位。

## 2. 伪共享 (False Sharing) 性能灾难
- 当两个独立的线程分别运行在不同的 CPU 核心上，且分别修改两个**逻辑上无关但物理地址紧挨在一起、落入同一个 64 字节缓存行内**的独立变量 $A$ 和 $B$ 时。
- MESI 缓存一致性协议会强制使对方核心的 Cache Line 处于 Invalid 状态，导致两个核心反复从主内存重新加载缓存行，系统总线被锁死，多线程并发性能甚至不如单线程。

## 3. 规避对齐
通过对高并发共享变量执行内存填充（Padding）或使用编译器对齐指令（如 `alignas(64)`），保证独立变量分布在不同的缓存行中。"""),

    ("numa_architecture_affinity", "NUMA 架构内存不均匀访问与 CPU 亲和性 (Affinity) 绑定", """# NUMA 架构内存不均匀访问与 CPU 亲和性 (Affinity) 绑定

## 1. SMP 对称多处理 vs NUMA 非一致性内存访问
- **SMP (Symmetric Multi-Processing)**：所有 CPU 共享同一条内存总线，随着 CPU 核心数增多总线成为严重瓶颈。
- **NUMA**：将 CPU 与内存划分为多个 Node。CPU 访问本地 Node 内存延迟极低；访问远端 Node 内存需要经过跨芯片互联总线（如 QPI/UPI），延迟高出 2~3 倍。

## 2. CPU 亲和性 (CPU Affinity) 与隔离 (numactl)
在高并发低延迟中间件中，若线程被操作系统调度器频繁调度到跨 Node 的核心上执行，会导致频繁的跨 Node 访存和缓存失效率。
- 使用 `pthread_setaffinity_np` 绑定工作线程到特定 CPU 核。
- 使用 `numactl --interleave` 或 `--cpunodebind` 指定内存分配策略。"""),

    ("context_switch_cost_registers", "进程/线程上下文切换 (Context Switch) 的微观物理开销", """# 进程/线程上下文切换 (Context Switch) 的微观物理开销

## 1. 直接开销 (Direct Overhead)
- 保存与恢复通用寄存器、栈指针（ESP/RSP）、指令指针（EIP/RIP）。
- 切换内核栈与用户栈。
- 进程级切换还需要刷新页表寄存器（CR3），导致整个 MMU 的虚拟地址映射失效。

## 2. 间接开销 (Indirect Overhead - 性能大头)
- **TLB 全面失效**：切换后初期的访存引发海量 TLB Miss。
- **CPU L1/L2 Cache 局部性破坏**：原进程填充在 CPU 高速缓存中的热数据被新进程的指令和数据驱逐，导致新进程在几千个周期内频繁发生 Cache Miss。"""),

    ("linux_cfs_scheduler_vruntime", "Linux CFS 完全公平调度器与虚拟运行时间 (vruntime)", """# Linux CFS 完全公平调度器与虚拟运行时间 (vruntime)

## 1. 完全公平的核心哲学
CFS 放弃了传统的固定时间片分配机制，采用红黑树动态维护就绪队列中所有任务的虚拟运行时间（`vruntime`）。

## 2. 虚拟运行时间换算公式
$$\\Delta vruntime = \\Delta exec\\_time \\times \\frac{\\text{NICE\\_0\\_LOAD}}{\\text{se}->load.weight}$$
- 权重越高的进程（Nice 值越小，优先级越高），其物理运行时间换算出的 $vruntime$ 增长得越慢。
- CFS 调度器在每次时钟中断或调度时，**永远优先选择红黑树最左侧（$vruntime$ 最小）的节点执行**，从而在宏观上达到极度精确的时间片加权分配。"""),

    ("rcu_read_copy_update_synchronization", "RCU (Read-Copy-Update) 读写同步机制与宽限期 (Grace Period)", """# RCU (Read-Copy-Update) 读写同步机制与宽限期 (Grace Period)

## 1. 读极多、写极少场景的传统锁开销
在网络路由表、文件描述符表等场景中，99.9% 是读操作。即使采用读写锁（rwlock），读锁的大量加锁解锁也会产生原子 CAS 指令总线竞争。

## 2. RCU 核心三大操作
- **Read-Lock-Free**：读操作无需任何锁，直接读取数据指针，开销几乎为 0。
- **Copy & Update**：写者要修改数据时，先复制一份数据副本，在副本上进行修改，然后通过原子指针替换将全局指针指向新数据。
- **Grace Period (宽限期)**：旧数据不能立即释放，必须等待所有在指针替换之前开始读取的读线程全部离开临界区（经历一个宽限期）后，才安全释放旧内存。"""),

    ("hard_irq_vs_softirq_ksoftirqd", "Linux 中断体系：上半部硬中断 (Hard IRQ) 与下半部软中断 (SoftIRQ)", """# Linux 中断体系：上半部硬中断 (Hard IRQ) 与下半部软中断 (SoftIRQ)

## 1. 为什么需要拆分上半部与下半部
硬件中断（如网卡收包）发生时，CPU 必须立即响应并关中断执行。如果处理耗时过长，会导致后续的其他硬件中断丢失，系统响应卡死。

## 2. 职责分离
- **上半部 (Top Half / 硬中断)**：快速响应硬件信号，做最简单且紧急的事（如将网卡描述符环的数据指针取下、清硬件状态寄存器），随后发起软中断并立即开中断返回。
- **下半部 (Bottom Half / SoftIRQ)**：以开中断状态异步执行繁重任务（如 TCP 报文校验、组装、路由与进程唤醒）。
- **ksoftirqd 内核守护线程**：当软中断过于频繁积压时，交由专用的 `ksoftirqd/X` 线程调度执行，避免饿死普通用户空间进程。"""),

    ("memory_barriers_relaxed_consistency", "内存屏障 (Memory Barrier) 与弱内存模型重排序", """# 内存屏障 (Memory Barrier) 与弱内存模型重排序

## 1. 乱序执行的诱因
现代超标量 CPU 为了提升吞吐量，允许执行指令级乱序（Out-of-Order Execution）；同时编译器也会进行指令重排优化；CPU 写缓冲区（Store Buffer）使得写操作不会立即同步到高速缓存。

## 2. 三种硬件内存屏障
- **读屏障 (Read Memory Barrier, rmb)**：确保屏障之前的所有 Load 指令全部先于屏障之后的所有 Load 指令完成。
- **写屏障 (Write Memory Barrier, wmb)**：确保屏障之前的 Store 操作先于屏障之后的 Store 操作刷新至缓存，保证写顺序。
- **全屏障 (Full Barrier, mfence)**：强制序列化所有的读写操作，彻底禁止跨屏障指令重排。"""),

    ("vfs_dentry_inode_architecture", "Linux VFS 虚拟文件系统抽象：dentry, inode 与 file 结构", """# Linux VFS 虚拟文件系统抽象：dentry, inode 与 file 结构

## 1. VFS 四大核心对象
- **superblock (超级块)**：代表一个已挂载的具体文件系统（如 ext4, xfs），保存全局元数据和块大小。
- **inode (索引节点)**：代表物理磁盘上的一个具体文件对象，记录权限、大小、创建时间及数据块指针，不包含文件名。
- **dentry (目录项)**：代表路径中的一个层级分段（如 `/usr/bin/cat` 中的每一段），将文件名与具体的 inode 编号进行绑定。VFS 维护内存 `dcache` 树加速路径解析。
- **file**：代表进程已经打开的文件句柄，维护当前的读取偏移量（`f_pos`）与打开模式。"""),

    ("network_namespace_veth_pair", "容器网络底层基石：Network Namespace 与 veth-pair 管道", """# 容器网络底层基石：Network Namespace 与 veth-pair 管道

## 1. Network Namespace 逻辑隔离
Linux 内核通过 Namespace 实现轻量级容器隔离。独立的 Net Namespace 拥有各自独立的路由表、iptables 规则链、网络设备列表与 Socket 端口空间。

## 2. veth-pair 虚拟以太网对
- 像一根双向连通的“虚拟网线”，必须成对创建。
- 从一端发出的数据包会被内核无缝重定向从另一端直接接收。
- **容器互联模式**：veth 的一端放置在容器内部重命名为 `eth0`；另一端插在宿主机的虚拟网桥（如 `docker0` 或 `cni0`）上，配合网桥广播与路由转发实现跨容器通信。"""),

    ("dma_ring_buffer_sk_buff", "网卡驱动接收数据全景：DMA 环形缓冲区与 NAPI 轮询", """# 网卡驱动接收数据全景：DMA 环形缓冲区与 NAPI 轮询

## 1. 网卡初始化与 RX Ring Buffer
驱动在启动时预先分配一块连续的物理内存作为环形缓冲区（Ring Buffer），将每个描述符初始化指向一个预分配的 `sk_buff` 物理地址。

## 2. NAPI 混合轮询机制
传统每个数据包一次硬中断会在高速万兆网络下使 CPU 彻底瘫痪。
- 数据包到达，网卡通过 DMA 将报文写入物理内存，触发一次硬中断。
- 驱动在上半部关闭网卡的中断功能，将设备加入 CPU 的轮询列表，触发软中断 `NET_RX_SOFTIRQ`。
- 下半部通过 `napi_poll()` 在一次循环中连续批量读取并处理几十个数据包，随后重新开启硬中断，在高负载下显著提升吞吐。"""),

    ("tcp_fast_open_tfo", "TCP Fast Open (TFO): 消除三次握手 1-RTT 延迟", """# TCP Fast Open (TFO): 消除三次握手 1-RTT 延迟

## 1. 传统握手的 1-RTT 延迟惩罚
客户端在发出 SYN 后的整整一个往返时间（1-RTT）内不能发送任何实际的 HTTP Payload，在短连接移动网络中极大拉长首包时间。

## 2. TFO 的加密 Cookie 验证机制
- **首次连接请求**：Client 发送带 TFO 选项的 SYN，Server 验证后在 SYN-ACK 中颁发一个加密的 TFO Cookie。
- **后续连接极速传输**：Client 再次连接时，直接在 **SYN 报文中携带该 TFO Cookie 和真实的 HTTP 业务数据**。
- Server 验证 Cookie 合法后，立即将数据送交应用层，在握手尚未完全完成之前即可开始处理业务逻辑，实现真正的 0-RTT 首包加速。"""),

    ("tcp_bbr_pacing_rate_gain", "BBR 算法状态机与 Pacing 速率平滑发送", """# BBR 算法状态机与 Pacing 速率平滑发送

## 1. 传统 TCP 的突发冲击 (Bursty Traffic)
传统拥塞控制通常在收到一个 ACK 时突发性发送 2~3 个数据包，导致中间路由器队列瞬间激增。

## 2. BBR 四大状态机
- **Startup**：以指数级倍增速率迅速打满管道带宽。
- **Drain**：排空在 Startup 阶段意外在中间路由器积攒的多余队列。
- **ProbeBW (主稳态)**：周期性微调增益系数（Pacing Gain，如在 1.25 和 0.75 之间巡航轮转），动态探索可用带宽。
- **ProbeRTT**：将在网报文降至极低水平，持续 200ms，精准捕获真实的链路最小传播延迟。

## 3. Pacing 机制
内核通过发包调度器（如 FQ）将数据包按照精确的微秒级时间间隔均匀打散发出，彻底抚平网络突刺。"""),

    ("tcp_syn_queue_accept_queue_leak", "TCP 半连接队列 (syns queue) 与全连接队列 (accept queue) 溢出诊断", """# TCP 半连接队列与全连接队列溢出诊断

## 1. 队列满载的物理表现
- **半连接队列溢出**：收到 SYN 时若队列满且未开启 SYN Cookie，Server 直接丢弃 SYN，Client 超时重传。
- **全连接队列溢出**：三次握手完成，但上层应用（如 Java, Nginx）由于 CPU 繁忙未及时执行 `accept()`。
  - 由 `/proc/sys/net/ipv4/tcp_abort_on_overflow` 控制：若为 0，Server 直接丢弃最后的 ACK，假装未收到，迫使 Client 重发；若为 1，直接向 Client 发送 RST 强制重置连接。

## 2. 诊断命令
- `ss -lnt` 中的 `Send-Q` 代表全连接队列的最大容量（`backlog` 参数），`Recv-Q` 代表当前等待 `accept()` 的连接数。
- 当 `Recv-Q > Send-Q` 时，意味着全连接队列已彻底被打满，必须紧急调大参数或优化应用层消费速度。""")
]
