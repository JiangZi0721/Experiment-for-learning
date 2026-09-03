# PostgreSQL MVCC 实现机制与 VACUUM 垃圾回收

## 1. 行头元组与可见性标记
PostgreSQL 在每个元组（Tuple）头信息中记录：
- `xmin`：创建该元组的事务 ID。
- `xmax`：删除或更新该元组的事务 ID（更新在物理上等同于“标记旧元组 xmax + 插入新元组”）。
- 当事务查询时，通过比对自身活跃事务快照（Snapshot）与元组的 xmin/xmax 判断可见性。

## 2. 膨胀 (Table Bloat) 与 VACUUM
- 物理就地保留死元组（Dead Tuples）会导致表和索引膨胀。
- **VACUUM**：扫描表并回收死元组空间，建立空闲空间映射（FSM）。
- **VACUUM FULL**：重写整张表释放磁盘给操作系统，需要获取排他表锁。
