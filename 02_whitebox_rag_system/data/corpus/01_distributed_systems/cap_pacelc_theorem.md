# CAP 定理与 PACELC 理论在生产架构中的抉择

## 1. CAP 定理重审
- **Consistency (强一致性)**：所有节点在同一时刻看到相同的数据。
- **Availability (高可用性)**：非故障节点必须对请求做出非错误响应。
- **Partition Tolerance (分区容忍性)**：网络断开时系统仍能工作。
由于分布式网络分区是物理必然（P 必须满足），系统只能在 CP（如 HBase, etcd）和 AP（如 Cassandra, CouchDB）中权衡。

## 2. PACELC 扩展理论
Daniel Abadi 指出 CAP 仅解释了分区（Partition）时的行为。PACELC 补全了常规无分区情况：
- **如果存在分区 (P)**：选择可用性 (A) 还是强一致性 (C)？
- **否则正常情况下 (Else)**：选择低延迟 (Latency, L) 还是强一致性 (Consistency, C)？
例如 MongoDB 通常属于 PC/EC（常态选延迟，分区选一致），而 DynamoDB 属于 PA/EL。
