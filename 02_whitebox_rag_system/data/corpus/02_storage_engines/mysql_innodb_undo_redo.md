# MySQL InnoDB 事务日志：Undo Log 与 Redo Log 的双剑合璧

## 1. Redo Log (重做日志) 保证持久性 (D)
- 物理-逻辑日志，记录页级别的物理修改（如“对第 5 号页偏移 100 写入数据”）。
- 循环覆盖写（Ring Buffer），包含 `checkpoint` 和 `write_pos`。
- 脏页依靠后台线程异步刷盘，崩溃恢复依靠 Redo Log 补全。

## 2. Undo Log (回滚日志) 保证原子性 (A) 与隔离性 (I)
- 逻辑日志，记录与操作相反的逆向 SQL（如 INSERT 对应 DELETE）。
- 为 MVCC 提供多版本读取链路：通过回滚指针（`roll_ptr`）将历史版本的 Undo Record 串成一条 Undo 链，供不同活跃视图（Read View）溯源读取。
