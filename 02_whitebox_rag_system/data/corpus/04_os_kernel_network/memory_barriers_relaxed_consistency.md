# 内存屏障 (Memory Barrier) 与弱内存模型重排序

## 1. 乱序执行的诱因
现代超标量 CPU 为了提升吞吐量，允许执行指令级乱序（Out-of-Order Execution）；同时编译器也会进行指令重排优化；CPU 写缓冲区（Store Buffer）使得写操作不会立即同步到高速缓存。

## 2. 三种硬件内存屏障
- **读屏障 (Read Memory Barrier, rmb)**：确保屏障之前的所有 Load 指令全部先于屏障之后的所有 Load 指令完成。
- **写屏障 (Write Memory Barrier, wmb)**：确保屏障之前的 Store 操作先于屏障之后的 Store 操作刷新至缓存，保证写顺序。
- **全屏障 (Full Barrier, mfence)**：强制序列化所有的读写操作，彻底禁止跨屏障指令重排。
