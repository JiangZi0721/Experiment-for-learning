# WAL 预写日志机制与 ARIES 崩溃恢复算法

## 1. WAL (Write-Ahead Logging) 核心铁律
**在任何脏数据页（Dirty Page）被刷入磁盘之前，必须先将对应修改的日志记录物理写入磁盘。**
保证即使系统在刷脏页时突发停电，重启后也能依靠 WAL 完整重建丢失的内存修改。

## 2. ARIES 恢复算法三大阶段
- **分析阶段 (Analysis Phase)**：从最近的检查点（Checkpoint）正向扫描 WAL，找出崩溃瞬间活跃的“未决事务表”与“脏页表”。
- **重做阶段 (Redo Phase)**：从最早未刷盘的脏页对应的日志序列号（LSN）开始，正向重放所有日志（包括崩溃前未提交事务的修改），重现崩溃瞬间的数据库历史状态（Repeating History）。
- **撤销阶段 (Undo Phase)**：逆向扫描 WAL，回滚所有在崩溃时未提交的活跃事务，写入补偿日志（Compensation Log Record, CLR），防止回滚过程再次宕机死循环。
