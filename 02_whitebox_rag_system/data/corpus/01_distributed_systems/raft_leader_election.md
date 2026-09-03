# Raft 领导者选举与心跳机制

## 1. 核心机制与角色状态机
Raft 将节点状态分为三种：Follower、Candidate 和 Leader。
- **Follower**：完全被动，接收来自 Leader 的心跳（AppendEntries RPC）或来自 Candidate 的投票请求（RequestVote RPC）。若选举超时（Election Timeout，通常为 150-300ms 随机值）未收到心跳，则自动转为 Candidate。
- **Candidate**：增加当前任期号（Current Term），为自己投一票，并向集群内其他所有节点并发广播 RequestVote RPC。
- **Leader**：获得集群超过半数（Quorum = n/2 + 1）节点的肯定投票后上位，立即向所有节点发送空 AppendEntries 作为心跳，压制其他节点的选举时钟。

## 2. 选举安全性与 Term 单调递增
- **Term 逻辑时钟**：Term 在 Raft 中充当逻辑时钟，解决脑裂（Split-Brain）与过期 Leader 问题。当节点收到比自己小的 Term 请求时直接拒绝；收到比自己大的 Term 时立即降级为 Follower。
- **选票唯一性约束**：在任何一个确定的 Term 内，每个节点最多只能投出一票（First-come, first-served），保证了一个 Term 内至多产生一个 Leader。
- **日志新旧比较约束 (Election Restriction)**：Candidate 的日志必须至少和投票者一样新（先比 Log Last Term，若相同比 Log Last Index），否则投票者必须拒绝投票。这保证了新 Leader 必然包含所有已经提交的日志条目。
