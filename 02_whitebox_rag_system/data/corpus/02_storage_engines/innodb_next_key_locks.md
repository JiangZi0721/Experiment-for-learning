# InnoDB 行锁演进：Record Lock, Gap Lock 与 Next-Key Lock

## 1. 三种锁的物理空间范围
- **Record Lock (记录锁)**：仅锁定索引项本身，防止其他事务并发修改该索引。
- **Gap Lock (间隙锁)**：锁定两个索引记录之间的开区间，防止其他事务在该间隙插入新数据。
- **Next-Key Lock**：Record Lock 与其前面的 Gap Lock 的组合（左开右闭区间 $(a, b]$），是 Repeatable Read 隔离级别下的默认锁算法。

## 2. 彻底消灭幻读 (Phantom Read)
在可重复读隔离级别下，Next-Key Lock 锁定了查询范围内的所有间隙，阻止了并发事务在此范围内 `INSERT` 产生“多出来的幽灵行”，在加锁读场景下完全消除了幻读。
