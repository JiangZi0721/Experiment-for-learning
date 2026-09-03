# Linux 虚拟内存系统与四级页表映射

## 1. 虚拟地址空间与 MMU
- 现代 64 位操作系统（如 x86_64）使用 48 位虚拟地址，划分为用户空间（User Space, $0\sim 0x00007FFFFFFFFFFF$）和内核空间（Kernel Space, $0xFFFF800000000000\sim 0xFFFFFFFFFFFFFFFF$）。
- 内存管理单元（MMU）负责将虚拟地址（VA）翻译为物理内存地址（PA）。

## 2. 四级页表结构 (PGD, P4D, PUD, PMD, PTE)
为了避免单级线性页表占用几百 GB 连续内存，Linux 采用多级稀疏树状页表：
1. **PGD (Page Global Directory)**：顶层全局页目录，CR3 寄存器存放其基地址。
2. **P4D (Page 4th Directory)**：支持 5 级分页的预留层。
3. **PUD (Page Upper Directory)**：页上级目录。
4. **PMD (Page Middle Directory)**：页中间目录。
5. **PTE (Page Table Entry)**：页表项，映射到真正的 4KB 物理页框（Page Frame），包含 Present, R/W, User/Supervisor 等控制标志。
