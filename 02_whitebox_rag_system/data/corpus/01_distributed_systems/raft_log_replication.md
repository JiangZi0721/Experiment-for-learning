# Raft 日志复制与一致性检查

## 1. 日志条目结构与追加流程
Leader 接收到客户端的写请求后，执行两阶段提交：
1. **追加本地日志**：Leader 将操作封装为日志条目（包含 Term 和 Index），写入本地 WAL。
2. **广播复制**：Leader 并发向所有 Follower 发送 AppendEntries RPC。
3. **安全提交**：当且仅当该条目已被复制到多数派（Quorum）节点上，Leader 才将其标记为已提交（Committed），并应用到状态机（State Machine），最后向客户端返回成功响应。

## 2. 日志对齐与回退机制 (Log Matching Property)
如果 Follower 与 Leader 日志发生冲突（例如 Follower 曾经是旧 Leader 且产生未提交孤儿日志）：
- Leader 为每个 Follower 维护 `nextIndex`（下一个要发送的日志索引）和 `matchIndex`（已知已同步的最大索引）。
- AppendEntries 携带 `prevLogIndex` 和 `prevLogTerm`。Follower 校验本地对应位置的日志，若不匹配则拒绝该 RPC。
- 一旦被拒绝，Leader 将该 Follower 的 `nextIndex` 递减并重发，直到找到双方日志完全一致的点，随后 Leader 强制用自己的日志覆盖 Follower 后续冲突的全部条目。
