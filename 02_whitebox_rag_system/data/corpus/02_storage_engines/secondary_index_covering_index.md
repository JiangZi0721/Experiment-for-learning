# 二级索引与覆盖索引 (Covering Index) 优化原理

## 1. 回表 (Index Lookup Back) 的性能损耗
通过二级索引定位到主键后，需要拿着主键再次遍历主键聚簇索引 B+ 树，产生额外的随机 I/O。

## 2. 覆盖索引的最佳实践
- 如果一个索引包含了查询所需的所有字段（例如联合索引 `(user_id, status, created_at)`），SQL 查询 `SELECT status, created_at FROM orders WHERE user_id = ?` 只需扫描该二级索引树即可直接返回结果，完全无需回表。
- 极大压制读放大，是高并发查询调优的基石。
