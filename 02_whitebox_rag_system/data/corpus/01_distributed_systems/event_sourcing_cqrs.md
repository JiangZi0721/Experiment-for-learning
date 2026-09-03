# 事件溯源 (Event Sourcing) 与 CQRS 架构解耦

## 1. 事件溯源 (Event Sourcing)
- 放弃传统关系型数据库将“最终状态”直接覆写旧数据的做法。
- 将系统的一切状态变更记录为不可变的原子事件流（Append-Only Event Store）。
- 业务当前状态是历史所有事件从初始状态按序重放（Replay）推导出来的投影（Projection）。

## 2. CQRS 命令查询职责分离
- **Command 侧**：负责处理写请求、校验业务约束并写入事件存储，强调强一致性。
- **Query 侧**：订阅事件总线，异步将事件增量物化为针对特定查询优化的只读视图（如 ElasticSearch、Redis），提供超高并发读能力。
