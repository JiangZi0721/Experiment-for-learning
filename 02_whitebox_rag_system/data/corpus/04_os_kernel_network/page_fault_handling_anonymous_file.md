# Linux 缺页异常处理机制：匿名页 vs 文件页

## 1. 延时分配机制 (Lazy Allocation)
当用户调用 `malloc()` 或 `mmap()` 时，内核仅仅在进程的 `vm_area_struct` 中记录一段虚地址范围，并不实际分配物理内存。只有当进程第一次对该地址进行读写时，CPU 触发 14 号异常——缺页中断（Page Fault）。

## 2. 两大类缺页路径
- **匿名页缺页 (Anonymous Page Fault)**：如堆（Heap）、栈（Stack）和私有写拷贝内存。内核直接从物理空闲伙伴系统（Buddy System）分配一个清零页建立映射。
- **文件页缺页 (File-backed Page Fault)**：如可执行代码段、通过 `mmap` 映射的文件。内核必须先在 Page Cache 中查找；若未命中，发起磁盘驱动器 I/O 将文件扇区加载进页缓存，再建立 PTE 映射。
