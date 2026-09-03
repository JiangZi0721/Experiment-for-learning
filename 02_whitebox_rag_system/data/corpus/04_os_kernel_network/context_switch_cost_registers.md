# 进程/线程上下文切换 (Context Switch) 的微观物理开销

## 1. 直接开销 (Direct Overhead)
- 保存与恢复通用寄存器、栈指针（ESP/RSP）、指令指针（EIP/RIP）。
- 切换内核栈与用户栈。
- 进程级切换还需要刷新页表寄存器（CR3），导致整个 MMU 的虚拟地址映射失效。

## 2. 间接开销 (Indirect Overhead - 性能大头)
- **TLB 全面失效**：切换后初期的访存引发海量 TLB Miss。
- **CPU L1/L2 Cache 局部性破坏**：原进程填充在 CPU 高速缓存中的热数据被新进程的指令和数据驱逐，导致新进程在几千个周期内频繁发生 Cache Miss。
