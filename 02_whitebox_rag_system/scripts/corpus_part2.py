# -*- coding: utf-8 -*-
"""02_storage_engines 25 topics"""

STORAGE_ENGINES = [
    ("lsm_tree_architecture", "LSM-Tree 核心架构与三大放大效应 (WAF/RAF/SAF)", """# LSM-Tree 核心架构与三大放大效应

## 1. LSM-Tree 存储分层与写入流水线
LSM-Tree (Log-Structured Merge-tree) 采用“追加写”换取高吞吐写入性能：
1. **内存 MemTable**：写入首先记录 WAL 保证持久性，然后插入内存跳表（Skiplist）。
2. **只读 Immutable MemTable**：MemTable 达到阈值后冻结为只读，后台异步线程执行 Flush。
3. **分层磁盘 SSTable**：Flush 落盘形成有序字符串表（Sorted String Table, SSTable）。

## 2. 三大放大效应 (Trade-offs)
- **写放大 (Write Amplification, WAF)**：实际写入物理磁盘的字节数与用户逻辑写入字节数的比值。Compaction 过程反复读取并重写数据，WAF 通常高达 10~30。
- **读放大 (Read Amplification, RAF)**：一次点查可能需要遍历 MemTable、Bloom Filter 以及多个层级（Levels）的 SSTable，导致数十次磁盘 I/O。
- **空间放大 (Space Amplification, SAF)**：由于数据覆写与删除采用追加墓碑标记（Tombstone），旧版本与无效数据在 Compaction 前占用额外磁盘空间。"""),

    ("sstable_bloom_filter", "SSTable 物理存储格式与布隆过滤器加速", """# SSTable 物理存储格式与布隆过滤器加速

## 1. SSTable 二进制物理布局
SSTable 通常划分为固定大小的数据块（Data Block，通常 4KB~64KB）：
- **Data Blocks**：按键有序存储的键值对，通常采用前缀压缩（Prefix Encoding）。
- **Index Block**：记录每个 Data Block 的最大键与文件偏移量，允许在内存中进行二分查找。
- **Filter Block**：包含整个 SSTable 的布隆过滤器（Bloom Filter）位图。
- **Footer**：位于文件末尾，记录 Index Block 和 Meta Index Block 的偏移量与 Magic Number。

## 2. 布隆过滤器的误报率与剪枝
在读取未命中某 SSTable 时，布隆过滤器能在 $O(1)$ 时间内确定“该 Key 绝对不存在”，从而跳过读取该文件磁盘 I/O。
- 公式：最佳位图大小 $m = -\\frac{n \\ln p}{(\\ln 2)^2}$，通常每个 Key 分配 10 bit 时误报率约为 1%。"""),

    ("memtable_skiplist_concurrency", "MemTable 内存跳表 (Skiplist) 并发机制", """# MemTable 内存跳表 (Skiplist) 并发机制

## 1. 为何选用跳表而非红黑树
- **并发锁竞争更低**：红黑树在插入节点时会触发全局旋转平衡，需要持有大范围树锁；跳表只需局部修改前驱/后继指针，天然支持细粒度 CAS 无锁并发（Lock-Free Concurrent SkipList）。
- **范围查询 (Range Scan) 极快**：跳表底层为全量有序双向链表，范围扫描只需移动指针。

## 2. 概率平衡与层高分配
- 跳表通过掷硬币机制（概率 $p=1/4$ 或 $1/2$）决定新节点的层高，时间复杂度期望为 $O(\\log n)$。
- 内存屏障（Memory Barrier）保证并发读线程在无锁遍历时不会读到悬挂指针。"""),

    ("compaction_size_tiered_vs_leveled", "Compaction 压缩策略对比：Size-Tiered vs Leveled", """# Compaction 压缩策略对比：Size-Tiered vs Leveled

## 1. Size-Tiered Compaction (STCS)
- **机制**：当某一层积累了若干大小相似的 SSTable（如 4 个）时，将它们合并为一个更大的 SSTable 存入下一层。
- **优势**：写放大较小（WAF 约为 5~10），适合写极其密集的场景。
- **劣势**：空间放大极大（临时需要预留 100% 额外磁盘空间），点查性能随 SSTable 数量增多而退化。

## 2. Leveled Compaction (LCS)
- **机制**：每一层限制总容量（如 $L_{i+1} = 10 \\times L_i$）。除 $L_0$ 外，每一层的各个 SSTable 的 Key 范围绝对互斥且全局有序。
- **优势**：点查延迟极低（单层最多命中一个文件），空间放大极小（通常低于 1.2）。
- **劣势**：写放大严重（WAF 达 20~40），频繁触发下层重叠文件重写。"""),

    ("b_plus_tree_node_layout", "B+ Tree 节点内部布局与高扇出 (Fanout) 设计", """# B+ Tree 节点内部布局与高扇出 (Fanout) 设计

## 1. 结构特征
- **高扇出度**：内部节点只存放键（Key）和子节点指针，不存数据载荷（Payload）。一个 16KB 页可容纳数百个键，树高通常仅为 3~4 层。
- **叶子节点全量数据与双向链接**：叶子节点包含全部真实数据或主键索引，并通过双向链表相连，范围扫描只需从首叶子节点遍历链表。

## 2. 磁盘预读与局部性原理
- 内部节点高度常驻内存缓冲池（Buffer Pool），根节点常驻内存，点查通常只需 1 次叶子磁盘 I/O。
- 缺点：写操作涉及就地更新（In-Place Update），导致大量随机写（Random Write I/O），固态硬盘写寿命与吞吐显著低于顺序写。"""),

    ("b_plus_tree_latch_crabbing", "B+ Tree 并发控制：蟹行加锁 (Latch Crabbing) 原理", """# B+ Tree 并发控制：蟹行加锁 (Latch Crabbing) 原理

## 1. 锁与闩锁的区别 (Lock vs Latch)
- **Lock (逻辑锁)**：事务级概念，用于保障事务 ACID，持续整个事务生命周期。
- **Latch (物理闩锁)**：线程级读写锁/互斥量，用于保护内存数据结构指针完整性，毫秒级即释放。

## 2. 蟹行加锁 (Latch Crabbing / Coupling) 机制
自顶向下遍历树：
1. 先获取父节点的 Latch。
2. 再获取子节点的 Latch。
3. **安全检查**：若子节点处于“安全状态”（对于读操作，总是安全；对于插入，子节点未满；对于删除，子节点元素足够不会触发合并），则**立即释放父节点的 Latch**。
4. 这种像螃蟹爬行一样的加锁释放交替，显著减少了树根节点的并发阻塞瓶颈。"""),

    ("wal_aries_recovery_algorithm", "WAL 预写日志机制与 ARIES 崩溃恢复算法", """# WAL 预写日志机制与 ARIES 崩溃恢复算法

## 1. WAL (Write-Ahead Logging) 核心铁律
**在任何脏数据页（Dirty Page）被刷入磁盘之前，必须先将对应修改的日志记录物理写入磁盘。**
保证即使系统在刷脏页时突发停电，重启后也能依靠 WAL 完整重建丢失的内存修改。

## 2. ARIES 恢复算法三大阶段
- **分析阶段 (Analysis Phase)**：从最近的检查点（Checkpoint）正向扫描 WAL，找出崩溃瞬间活跃的“未决事务表”与“脏页表”。
- **重做阶段 (Redo Phase)**：从最早未刷盘的脏页对应的日志序列号（LSN）开始，正向重放所有日志（包括崩溃前未提交事务的修改），重现崩溃瞬间的数据库历史状态（Repeating History）。
- **撤销阶段 (Undo Phase)**：逆向扫描 WAL，回滚所有在崩溃时未提交的活跃事务，写入补偿日志（Compensation Log Record, CLR），防止回滚过程再次宕机死循环。"""),

    ("postgresql_mvcc_vacuum", "PostgreSQL MVCC 实现机制与 VACUUM 垃圾回收", """# PostgreSQL MVCC 实现机制与 VACUUM 垃圾回收

## 1. 行头元组与可见性标记
PostgreSQL 在每个元组（Tuple）头信息中记录：
- `xmin`：创建该元组的事务 ID。
- `xmax`：删除或更新该元组的事务 ID（更新在物理上等同于“标记旧元组 xmax + 插入新元组”）。
- 当事务查询时，通过比对自身活跃事务快照（Snapshot）与元组的 xmin/xmax 判断可见性。

## 2. 膨胀 (Table Bloat) 与 VACUUM
- 物理就地保留死元组（Dead Tuples）会导致表和索引膨胀。
- **VACUUM**：扫描表并回收死元组空间，建立空闲空间映射（FSM）。
- **VACUUM FULL**：重写整张表释放磁盘给操作系统，需要获取排他表锁。"""),

    ("mysql_innodb_undo_redo", "MySQL InnoDB 事务日志：Undo Log 与 Redo Log 的双剑合璧", """# MySQL InnoDB 事务日志：Undo Log 与 Redo Log 的双剑合璧

## 1. Redo Log (重做日志) 保证持久性 (D)
- 物理-逻辑日志，记录页级别的物理修改（如“对第 5 号页偏移 100 写入数据”）。
- 循环覆盖写（Ring Buffer），包含 `checkpoint` 和 `write_pos`。
- 脏页依靠后台线程异步刷盘，崩溃恢复依靠 Redo Log 补全。

## 2. Undo Log (回滚日志) 保证原子性 (A) 与隔离性 (I)
- 逻辑日志，记录与操作相反的逆向 SQL（如 INSERT 对应 DELETE）。
- 为 MVCC 提供多版本读取链路：通过回滚指针（`roll_ptr`）将历史版本的 Undo Record 串成一条 Undo 链，供不同活跃视图（Read View）溯源读取。"""),

    ("columnar_storage_parquet_arrow", "列式存储物理格局：Apache Parquet 与 Apache Arrow 对比", """# 列式存储物理格局：Apache Parquet 与 Apache Arrow 对比

## 1. 行存 (OLTP) vs 列存 (OLAP)
- **行存**：同一行的数据物理连续，适合单行 CRUD、点查。但 OLAP 分析聚合查询时会产生海量无效列的磁盘 I/O。
- **列存**：同一列的数据连续存储，聚合 `SUM(salary)` 只需读取一列数据，磁盘扫描量骤减 90%。

## 2. Parquet (磁盘级存储)
- 基于 Dremel 嵌套结构的二进制文件格式。按 Row Group、Column Chunk 和 Page 分层，支持字典编码、Run-Length 编码（RLE）与 Snappy/ZSTD 极高压缩率。

## 3. Arrow (内存中向量化执行)
- 跨语言的标准内存列式布局。通过对齐的扁平缓冲区（Contiguous Buffer）实现零反序列化拷贝，配合 CPU SIMD 向量化指令实现极致计算加速。"""),

    ("buffer_pool_lru_k_algorithm", "数据库缓冲池管理：LRU-K 与 Clock Sweep 淘汰算法", """# 数据库缓冲池管理：LRU-K 与 Clock Sweep 淘汰算法

## 1. 传统 LRU 的缓存污染 (Buffer Pool Pollution)
全表扫描（Full Table Scan）会将大量仅读取一次的冷数据页加载进 LRU 链表头部，瞬间挤出频繁使用的高价值热数据页。

## 2. LRU-2 (LRU-K) 解决方案
- 维护两个队列：历史访问队列与缓存命中队列。
- 数据页首次访问只进入历史队列；只有在倒数第 $K$ 次被访问后，才提升到真正的缓存队列。
- 彻底过滤单次扫描数据，淘汰依据变为“倒数第 $K$ 次访问的时间距今最远”。

## 3. Clock Sweep 时钟算法
每个缓冲页维护一个使用标志位（Usage Bit）。扫描指针像时钟指针一样转动：若为 1 则清 0 放过；若已为 0 则直接淘汰并写入脏页，实现高并发无锁近似 LRU。"""),

    ("doublewrite_buffer_partial_page_write", "InnoDB Doublewrite Buffer 与页部分写入失效 (Partial Page Write)", """# InnoDB Doublewrite Buffer 与页部分写入失效

## 1. 问题的物理本质
- 数据库的一个数据页通常为 16KB，而底层操作系统的页是 4KB，物理磁盘扇区通常为 512B 或 4KB。
- 当向磁盘刷入 16KB 页时，若在写完前 4KB 时突发掉电，导致该页处于新旧交织的物理损坏状态（Partial Page Write / Torn Page）。
- **Redo Log 无法挽救**：Redo Log 记录的是增量物理偏移，若基础页本身校验和校验失败且数据错乱，无法在其上执行 Redo 运算。

## 2. 双写缓冲解决机制
- 在把脏页刷入真正的表空间前，先将其**顺序写**到共享表空间的 Doublewrite Buffer（顺序写开销极小）。
- 随后才将脏页离散写入各个数据文件。
- 若离散写崩溃损坏，恢复时可直接从 Doublewrite Buffer 中拷贝出完整副本重新执行恢复。"""),

    ("innodb_next_key_locks", "InnoDB 行锁演进：Record Lock, Gap Lock 与 Next-Key Lock", """# InnoDB 行锁演进：Record Lock, Gap Lock 与 Next-Key Lock

## 1. 三种锁的物理空间范围
- **Record Lock (记录锁)**：仅锁定索引项本身，防止其他事务并发修改该索引。
- **Gap Lock (间隙锁)**：锁定两个索引记录之间的开区间，防止其他事务在该间隙插入新数据。
- **Next-Key Lock**：Record Lock 与其前面的 Gap Lock 的组合（左开右闭区间 $(a, b]$），是 Repeatable Read 隔离级别下的默认锁算法。

## 2. 彻底消灭幻读 (Phantom Read)
在可重复读隔离级别下，Next-Key Lock 锁定了查询范围内的所有间隙，阻止了并发事务在此范围内 `INSERT` 产生“多出来的幽灵行”，在加锁读场景下完全消除了幻读。"""),

    ("hash_index_bitcask_model", "Append-Only 存储引擎：Bitcask 日志模型与哈希索引", """# Append-Only 存储引擎：Bitcask 日志模型与哈希索引

## 1. 结构极简主义
Bitcask 是 Riak 底层采用的高性能只追加键值存储模型：
- **磁盘存储**：所有写入操作直接追加写到只追加日志文件（Append-only Log）。更新和删除也仅仅是追加新值或追加墓碑标记。
- **内存哈希表 (KeyDir)**：内存中全量保存每个 Key 对应的最新文件 ID、文件偏移量（Offset）和数据长度。

## 2. 优劣分析
- **写性能极限**：纯顺序 I/O 写入。
- **读性能极高**：内存哈希查出偏移后，只需 1 次精准磁盘 Seeking。
- **致命缺点**：无法进行范围扫描（Range Query）；且内存必须能够装下全局所有的 Key。"""),

    ("write_stall_throttling_lsm", "LSM-Tree 的写停顿 (Write Stall) 与流量削峰", """# LSM-Tree 的写停顿 (Write Stall) 与流量削峰

## 1. 产生原因
当上层客户端写入吞吐远超过后台异步压缩（Compaction）的处理能力时：
- $L_0$ 层文件数急剧堆积。由于 $L_0$ 内部键范围重叠，文件过多会导致读性能断崖式下跌。
- MemTable 内存用尽，而后台 Flush 线程被磁盘 I/O 挤死，无法及时腾出新 MemTable。

## 2. 削峰与平滑节流
RocksDB 引入了 Write Stall 机制：
- **渐进降速**：当 $L_0$ 文件达到阈值时，人工休眠客户端写线程微秒级时间，平滑限制入口 QPS。
- **硬停顿 (Hard Stop)**：若堆积继续恶化，完全阻塞写请求，将所有磁盘带宽让渡给后台 Compaction 线程救火。"""),

    ("checkpoint_sharp_vs_fuzzy", "数据库检查点技术：Sharp Checkpoint 与 Fuzzy Checkpoint", """# 数据库检查点技术：Sharp Checkpoint 与 Fuzzy Checkpoint

## 1. 检查点的目的
限制 WAL 日志的无限膨胀，并在崩溃恢复时确定重做日志的安全起点，避免从第一条日志全量重放。

## 2. 尖锐检查点 (Sharp Checkpoint)
- 强制阻塞所有写事务，将当前缓冲池中所有的脏页全部物理同步写入磁盘，随后记录检查点。
- **缺点**：导致系统产生巨大的 I/O 尖刺和瞬时性能停顿，生产环境极少使用。

## 3. 模糊检查点 (Fuzzy Checkpoint)
- 系统周期性记录当前脏页列表中“最老脏页的 LSN”。
- 允许后台线程异步持续刷脏页，不阻塞前端事务。
- 崩溃恢复时，只需从该记录的最老 LSN 处开始重放日志，兼顾低延迟与快速恢复。"""),

    ("snapshot_isolation_write_skew", "快照隔离 (Snapshot Isolation) 与写偏斜 (Write Skew) 异常", """# 快照隔离 (Snapshot Isolation) 与写偏斜 (Write Skew) 异常

## 1. 快照隔离定义
事务在启动时获取一个全局数据快照。在此快照下，事务读到的所有数据均具有一致性，并发事务的后续修改对其不可见。
- 规则：**First-Committer-Wins**。若两个并发事务修改了同一个键，后提交的事务中止回滚。

## 2. 写偏斜 (Write Skew) 异常
快照隔离并非可串行化（Serializable），它无法阻止写偏斜：
- **典型案例：黑白球互换**。约束条件为“至少保留一个黑球”。
  - 事务 1 读到两个黑球，将球 A 改为白球并提交。
  - 事务 2 并发执行，也读到两个黑球，将球 B 改为白球并提交。
  - 最终两个黑球全部变为白球，系统完整性破坏。必须升级为真正的 Serializable 快照隔离（SSI）通过读写依赖图（SGT）检测环路。"""),

    ("encoding_dictionary_delta_bitpacking", "列存压缩算法：字典编码、差分编码与位打包", """# 列存压缩算法：字典编码、差分编码与位打包

## 1. 字典编码 (Dictionary Encoding)
对于低基数（Cardinality）列（如国家、状态）：
- 建立映射字典：`{"Beijing": 0, "Shanghai": 1, "Guangzhou": 2}`。
- 数据列只需存储紧凑的整数数组 `[0, 1, 0, 2]`，压缩率可达 80% 以上。

## 2. 差分编码 (Delta / Delta-of-Delta)
常用于时间戳与单调递增 ID：
- 存储与前一个数值的差值而非原始绝对值。
- 时间序列中，差值的差值（Delta-of-Delta）往往极其接近于 0，极其适合后续结合 RLE（游程编码）压缩。

## 3. 位打包 (Bit-Packing)
若一列整数的最大差值不超过 7，计算机不必为每个数分配 32/64 bit，只需用 3 bit 表示一个数，紧凑挤入字节流。"""),

    ("fsync_fdatasync_io_safety", "Linux 存储持久化调用：write, fsync 与 fdatasync 对比", """# Linux 存储持久化调用：write, fsync 与 fdatasync 对比

## 1. write() 的内核缓存欺骗
执行 `write()` 仅仅将数据写入内核的页缓存（Page Cache）即返回成功，数据并未真正落入物理驱动器介质。

## 2. fsync() vs fdatasync()
- **fsync(fd)**：强制将页缓存中的脏数据以及文件的**所有元数据（修改时间、大小、文件属性）**全部刷新到物理硬盘，并等待控制器写入确认。元数据与数据处于不同磁盘扇区，通常导致两次寻道写入。
- **fdatasync(fd)**：仅当元数据修改影响后续读取（如文件长度变大）时才同步元数据，若仅仅是修改时间戳变化则不刷元数据，从而减少了一次随机磁盘寻道，是高性能数据库写入 WAL 时的首选方案。"""),

    ("b_link_tree_concurrent_traversal", "B-link Tree 高并发遍历无锁演进", """# B-link Tree 高并发遍历无锁演进

## 1. B+ 树裂变时的死锁与宽范围加锁
标准 B+ 树节点分裂时需要同时锁定父节点和当前节点，并发冲突高。

## 2. B-link Tree 的右指针机制 (Lehman-Yao 算法)
- 在所有内部节点上也加入同层向右横向单向指针（High Key 和 Right Sibling Pointer）。
- 当节点分裂发生时，先创建新节点并连接右指针，再向父节点更新指针。
- 并发读线程遍历到旧节点时，若发现要查找的 Key 大于当前节点的 High Key，无需回退重试，直接顺着右指针跳到新节点继续读取，彻底实现了**读线程无需加锁**的超高并发并发度。"""),

    ("prefix_bloom_filter_rocksdb", "RocksDB 前缀布隆过滤器 (Prefix Bloom Filter) 优化", """# RocksDB 前缀布隆过滤器 (Prefix Bloom Filter) 优化

## 1. 传统全量布隆过滤器的局限
全量 Bloom Filter 只能针对完整 Key 进行存在性判定，对范围扫描（如 `iterator.Seek("user_1001_order_*")`）完全无效，每次范围扫描必须进入磁盘。

## 2. 前缀哈希与范围跳过
- 用户自定义前缀切分函数（如截取前 12 个字节作为前缀）。
- RocksDB 在构建 SSTable 时，针对 Key 的前缀单独计算哈希并放入 Prefix Bloom Filter。
- 执行前缀 Seek 操作时，若该前缀在 Bloom Filter 中判定不存在，直接跳过整个 SSTable，极大优化时序与多维查询。"""),

    ("index_organized_table_iot", "索引组织表 (IOT) 与堆表 (Heap Table) 物理选型", """# 索引组织表 (IOT) 与堆表 (Heap Table) 物理选型

## 1. 索引组织表 (如 MySQL InnoDB)
- 表中数据本身就是按照主键排序的一棵聚簇索引（Clustered Index）B+ 树。
- **二级索引代价**：二级索引叶子节点存放的是主键值，查非主键字段需要进行“回表”查询。主键过大将导致所有二级索引体积膨胀。

## 2. 堆表 (如 Oracle, PostgreSQL)
- 数据元组无序追加写入数据块中，物理地址由元组标识符（TID / RowID）决定。
- 所有索引（包括主键索引和二级索引）完全对等，其叶子节点均直接存储指向该数据块的 RowID。
- **优势**：更新非索引列无需移动物理位置；二级索引检索直接触达元组，免去回表代价。"""),

    ("zstd_vs_snappy_compression", "数据库压缩算法演进：Snappy 的低延迟 vs ZSTD 的极致压缩比", """# 数据库压缩算法演进：Snappy 的低延迟 vs ZSTD 的极致压缩比

## 1. Snappy：吞吐优先
- Google 开发，基于 LZ77 演进。不进行熵编码（Huffman 编码），追求极高解压缩吞吐（单核解压可达 500MB/s）。
- 常用于 LSM-Tree 的高层（如 $L_0, L_1$）或对查询延迟极度敏感的实时系统。

## 2. ZSTD (Zstandard)：压缩比与吞吐的现代巅峰
- Facebook 开发，结合了快速前缀匹配与有限状态熵（Finite State Entropy, FSE）。
- 提供 1 到 22 级的多档调节。在相同解压缩速度下，压缩比通常比 Snappy 高出 30%~50%，是底层冷数据（Cold Data）归档压缩的工业事实标准。"""),

    ("page_cache_vs_direct_io", "操作系统页缓存 (Page Cache) 与直接 I/O (O_DIRECT) 抉择", """# 操作系统页缓存 (Page Cache) 与直接 I/O (O_DIRECT) 抉择

## 1. 双重缓存困境 (Double Buffering)
数据库为了实现精确的并发控制和事务隔离，在用户空间维护了庞大的 Buffer Pool（如 128GB）。若通过标准 OS 调用读写，数据将在 Linux Page Cache 和 Buffer Pool 中各存一份，浪费一倍内存空间。

## 2. 直接 I/O (Direct I/O) 的绕行机制
- 打开文件时指定 `O_DIRECT` 标志，使读取和写入直接在用户内存与磁盘驱动器之间传输，完全绕过 Linux Page Cache。
- 允许数据库开发人员自主控制预读策略、脏页替换与顺序合并，彻底消除内核态上下文切换和不必要的内存拷贝。"""),

    ("secondary_index_covering_index", "二级索引与覆盖索引 (Covering Index) 优化原理", """# 二级索引与覆盖索引 (Covering Index) 优化原理

## 1. 回表 (Index Lookup Back) 的性能损耗
通过二级索引定位到主键后，需要拿着主键再次遍历主键聚簇索引 B+ 树，产生额外的随机 I/O。

## 2. 覆盖索引的最佳实践
- 如果一个索引包含了查询所需的所有字段（例如联合索引 `(user_id, status, created_at)`），SQL 查询 `SELECT status, created_at FROM orders WHERE user_id = ?` 只需扫描该二级索引树即可直接返回结果，完全无需回表。
- 极大压制读放大，是高并发查询调优的基石。""")
]
