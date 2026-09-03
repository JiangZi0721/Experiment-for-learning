# Multi-Paxos 优化与 Leader 租约机制

## 1. Classic Paxos 的性能瓶颈与活锁
Classic Paxos 达成单个值需要两轮 RPC（Prepare + Accept）。当多个 Proposer 并发提议且不断提高提案号时，极易陷入活锁（Livelock），导致系统无法收敛。

## 2. Multi-Paxos 的跳过 Phase 1 优化
- 通过选举产生唯一的 Leader（稳定的 Proposer）。
- Leader 针对整条实例序列统一执行一次 Phase 1，后续所有日志实例（Instance）只需执行 Phase 2（一轮 RPC），从而将通信延迟从 2-RTT 降低到 1-RTT。

## 3. Leader 租约 (Lease Read) 优化
- 为了防止网络分区导致旧 Leader 提供脏读（Stale Read），Leader 可以向 Quorum 获取具有时间有效期的租约（Lease）。
- 在租约有效期内，Follower 承诺不发起选举，Leader 确保自身拥有全局绝对领导权，可以直接读取本地状态机，避免每次读请求都走完整的共识日志复制。
