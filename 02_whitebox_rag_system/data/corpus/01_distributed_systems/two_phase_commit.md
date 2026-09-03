# 分布式事务二阶段提交 (2PC) 机制与死穴

## 1. 协议流程
- **Phase 1: 准备阶段 (Prepare)**：协调者（Coordinator）向所有参与者（Participants）发送 Prepare 请求。参与者执行本地事务、写 Redo/Undo 日志，锁定资源，但不提交，向协调者返回 VOTE_COMMIT 或 VOTE_ABORT。
- **Phase 2: 提交阶段 (Commit)**：协调者根据收集到的选票做决策。若所有参与者全票通过，协调者写 Commit 事务日志并向全员发送 Global_Commit；若有任一参与者投票失败或超时，协调者发送 Global_Abort。

## 2. 2PC 的致命死穴
1. **同步阻塞 (Synchronous Blocking)**：所有参与者在等待协调者决议期间持有数据库行锁，极大降低并发吞吐量。
2. **单点故障 (Single Point of Failure)**：协调者在 Phase 2 发出部分 Commit 后崩溃，参与者将陷入盲目阻塞状态（不清楚是否应该提交还是回滚）。
3. **数据不一致 (Data Inconsistency)**：网络分区发生时，部分节点收到 Commit，部分节点网络超时，导致部分提交、部分未决的严重脏数据。
