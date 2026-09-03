# -*- coding: utf-8 -*-
"""01_distributed_systems 25 topics"""

DISTRIBUTED_SYSTEMS = [
    ("raft_leader_election", "Raft 领导者选举与心跳机制", """# Raft 领导者选举与心跳机制

## 1. 核心机制与角色状态机
Raft 将节点状态分为三种：Follower、Candidate 和 Leader。
- **Follower**：完全被动，接收来自 Leader 的心跳（AppendEntries RPC）或来自 Candidate 的投票请求（RequestVote RPC）。若选举超时（Election Timeout，通常为 150-300ms 随机值）未收到心跳，则自动转为 Candidate。
- **Candidate**：增加当前任期号（Current Term），为自己投一票，并向集群内其他所有节点并发广播 RequestVote RPC。
- **Leader**：获得集群超过半数（Quorum = n/2 + 1）节点的肯定投票后上位，立即向所有节点发送空 AppendEntries 作为心跳，压制其他节点的选举时钟。

## 2. 选举安全性与 Term 单调递增
- **Term 逻辑时钟**：Term 在 Raft 中充当逻辑时钟，解决脑裂（Split-Brain）与过期 Leader 问题。当节点收到比自己小的 Term 请求时直接拒绝；收到比自己大的 Term 时立即降级为 Follower。
- **选票唯一性约束**：在任何一个确定的 Term 内，每个节点最多只能投出一票（First-come, first-served），保证了一个 Term 内至多产生一个 Leader。
- **日志新旧比较约束 (Election Restriction)**：Candidate 的日志必须至少和投票者一样新（先比 Log Last Term，若相同比 Log Last Index），否则投票者必须拒绝投票。这保证了新 Leader 必然包含所有已经提交的日志条目。"""),

    ("raft_log_replication", "Raft 日志复制与一致性检查", """# Raft 日志复制与一致性检查

## 1. 日志条目结构与追加流程
Leader 接收到客户端的写请求后，执行两阶段提交：
1. **追加本地日志**：Leader 将操作封装为日志条目（包含 Term 和 Index），写入本地 WAL。
2. **广播复制**：Leader 并发向所有 Follower 发送 AppendEntries RPC。
3. **安全提交**：当且仅当该条目已被复制到多数派（Quorum）节点上，Leader 才将其标记为已提交（Committed），并应用到状态机（State Machine），最后向客户端返回成功响应。

## 2. 日志对齐与回退机制 (Log Matching Property)
如果 Follower 与 Leader 日志发生冲突（例如 Follower 曾经是旧 Leader 且产生未提交孤儿日志）：
- Leader 为每个 Follower 维护 `nextIndex`（下一个要发送的日志索引）和 `matchIndex`（已知已同步的最大索引）。
- AppendEntries 携带 `prevLogIndex` 和 `prevLogTerm`。Follower 校验本地对应位置的日志，若不匹配则拒绝该 RPC。
- 一旦被拒绝，Leader 将该 Follower 的 `nextIndex` 递减并重发，直到找到双方日志完全一致的点，随后 Leader 强制用自己的日志覆盖 Follower 后续冲突的全部条目。"""),

    ("raft_safety_and_joint_consensus", "Raft 成员变更与联合共识", """# Raft 成员变更与联合共识 (Joint Consensus)

## 1. 单节点变更算法
为了避免集群在配置切换过程中出现两个独立的多数派，Raft 提出了单节点成员变更（Single-Server Changes）：
- 一次只允许增加或删除一个节点。
- 由于旧集群 $C_{old}$ 和新集群 $C_{new}$ 的大小仅相差 1，任意时刻 $C_{old}$ 的多数派与 $C_{new}$ 的多数派必然存在至少一个重叠节点，从而杜绝了双主脑裂。

## 2. 联合共识 (Joint Consensus)
对于多节点同时变更，采用两阶段联合配置：
- 第一阶段：Leader 提议联合配置 $C_{old,new}$。此时任意决策必须同时获得 $C_{old}$ 的多数派以及 $C_{new}$ 的多数派的批准。
- 第二阶段：当 $C_{old,new}$ 在两方多数派均已提交后，Leader 再提议新配置 $C_{new}$，完成平滑过渡。"""),

    ("paxos_classic_synod", "Classic Paxos 决议推演与两阶段提交", """# Classic Paxos 决议推演与两阶段提交

## 1. 角色定义
- **Proposer**：提案发起者，提出包含提案编号 $n$ 和提议值 $v$ 的提案。
- **Acceptor**：提案表决者，形成 Quorum 仲裁。
- **Learner**：学习者，感知已达成的决议。

## 2. 核心两阶段协议
### Phase 1: Prepare & Promise
- Proposer 选择提案号 $n$，向半数以上 Acceptor 发送 Prepare(n)。
- Acceptor 收到后，若 $n$ 大于它见过的所有提案号，则返回 Promise，承诺不再接受编号小于 $n$ 的提案，并附带它此前已接受的最大编号提案 $(n_{max}, v_{max})$。

### Phase 2: Accept & Accepted
- Proposer 收到多数派 Promise 后，检查所有回复。若有 Acceptor 曾接受过值，Proposer 必须将 $v$ 替换为回复中编号最大的 $v_{max}$；若无，则自由决定 $v$。随后广播 Accept(n, v)。
- Acceptor 收到 Accept 请求时，只要它此前没有承诺过只接受大于 $n$ 的提案，就必须批准该提案。"""),

    ("multi_paxos_lease", "Multi-Paxos 优化与 Leader 租约机制", """# Multi-Paxos 优化与 Leader 租约机制

## 1. Classic Paxos 的性能瓶颈与活锁
Classic Paxos 达成单个值需要两轮 RPC（Prepare + Accept）。当多个 Proposer 并发提议且不断提高提案号时，极易陷入活锁（Livelock），导致系统无法收敛。

## 2. Multi-Paxos 的跳过 Phase 1 优化
- 通过选举产生唯一的 Leader（稳定的 Proposer）。
- Leader 针对整条实例序列统一执行一次 Phase 1，后续所有日志实例（Instance）只需执行 Phase 2（一轮 RPC），从而将通信延迟从 2-RTT 降低到 1-RTT。

## 3. Leader 租约 (Lease Read) 优化
- 为了防止网络分区导致旧 Leader 提供脏读（Stale Read），Leader 可以向 Quorum 获取具有时间有效期的租约（Lease）。
- 在租约有效期内，Follower 承诺不发起选举，Leader 确保自身拥有全局绝对领导权，可以直接读取本地状态机，避免每次读请求都走完整的共识日志复制。"""),

    ("two_phase_commit", "分布式事务二阶段提交 (2PC) 机制与死穴", """# 分布式事务二阶段提交 (2PC) 机制与死穴

## 1. 协议流程
- **Phase 1: 准备阶段 (Prepare)**：协调者（Coordinator）向所有参与者（Participants）发送 Prepare 请求。参与者执行本地事务、写 Redo/Undo 日志，锁定资源，但不提交，向协调者返回 VOTE_COMMIT 或 VOTE_ABORT。
- **Phase 2: 提交阶段 (Commit)**：协调者根据收集到的选票做决策。若所有参与者全票通过，协调者写 Commit 事务日志并向全员发送 Global_Commit；若有任一参与者投票失败或超时，协调者发送 Global_Abort。

## 2. 2PC 的致命死穴
1. **同步阻塞 (Synchronous Blocking)**：所有参与者在等待协调者决议期间持有数据库行锁，极大降低并发吞吐量。
2. **单点故障 (Single Point of Failure)**：协调者在 Phase 2 发出部分 Commit 后崩溃，参与者将陷入盲目阻塞状态（不清楚是否应该提交还是回滚）。
3. **数据不一致 (Data Inconsistency)**：网络分区发生时，部分节点收到 Commit，部分节点网络超时，导致部分提交、部分未决的严重脏数据。"""),

    ("three_phase_commit", "三阶段提交 (3PC) 的超时与状态演进", """# 三阶段提交 (3PC) 的超时与状态演进

## 1. 拆分与引入超时机制
为了缓解 2PC 的同步阻塞问题，3PC 将准备阶段拆分为 CanCommit 和 PreCommit，并引入超时自愈机制：
1. **CanCommit**：询问参与者是否有能力执行事务，不锁资源。
2. **PreCommit**：若全员确认，协调者要求执行事务操作并记录 Undo/Redo 日志，进入预提交状态。
3. **DoCommit**：最终决议提交。

## 2. 参与者超时自动提交策略
在 3PC 中，一旦参与者进入 PreCommit 状态且等待协调者指令超时，参与者会默认假设多数派已经同意并**自动执行本地 Commit**。
- **残留缺陷**：若此时网络发生脑裂，协调者其实发出了 Abort，但超时分区的参与者擅自 Commit，依然会导致脑裂不一致。"""),

    ("saga_pattern_orchestration", "SAGA 分布式长事务与补偿机制", """# SAGA 分布式长事务与补偿机制

## 1. 核心理论模型
SAGA 将一个跨服务的长事务拆分为多个本地短事务序列 $T_1, T_2, ..., T_n$。每个事务 $T_i$ 都有一个对应的补偿事务 $C_i$。
- **正向成功**：按序执行 $T_1 \to T_2 \to ... \to T_n$。
- **故障回滚**：若 $T_k$ 执行失败，系统将反向依序执行已成功事务的补偿动作 $C_{k-1}, ..., C_2, C_1$。

## 2. 编排式 (Orchestration) vs 协同式 (Choreography)
- **协同式**：每个服务完成本地事务后发布事件，下一服务监听事件触发自身逻辑。缺点是依赖隐式，极易形成事件环路。
- **编排式**：由中央 SAGA 协调器（如 Temporal、Seata）通过状态机显式调用每个服务的接口，并负责失败时的补偿调度，具备更好的可观测性与隔离性。"""),

    ("tcc_transaction_model", "TCC 补偿型事务 (Try-Confirm-Cancel) 架构", """# TCC 补偿型事务 (Try-Confirm-Cancel) 架构

## 1. 阶段设计与业务隔离
TCC 是应用层两阶段提交的经典模式，要求业务提供三个接口：
- **Try**：业务检查与资源预留。例如扣减库存时，并非直接扣减可用库存，而是将库存划入“冻结字段”。
- **Confirm**：确认执行业务，直接使用 Try 阶段预留的资源，不进行任何额外的业务检查。要求幂等。
- **Cancel**：释放 Try 阶段预留的资源，将冻结字段还原为可用库存。同样要求幂等。

## 2. 异常控制：防悬挂、空回滚与幂等
- **空回滚**：Try 请求因网络丢包未到达，协调者发起 Cancel。Cancel 必须能识别出 Try 从未执行，直接返回成功而不真正扣减。
- **防悬挂 (Suspension)**：Cancel 请求先于延迟的 Try 请求到达。系统必须记录事务状态，后续迟到的 Try 到达时必须拒绝执行。"""),

    ("vector_clocks_dynamo", "向量时钟 (Vector Clocks) 与因果一致性", """# 向量时钟 (Vector Clocks) 与因果一致性

## 1. 物理时钟漂移难题与逻辑时钟演化
分布式系统中无法依靠单机物理时钟（Wall Clock）对并发事件定序（NTP 存在毫秒级时钟漂移 Clock Skew）。Lamport 逻辑时钟解决了全序关系，但无法区分并发事件。

## 2. 向量时钟数学定义
一个含有 $N$ 个节点的系统，每个节点维护向量 $V[1..N]$：
1. 节点 $i$ 产生本地事件时，自增 $V[i] = V[i] + 1$。
2. 节点 $i$ 发送消息携带自己的向量 $V_i$。
3. 节点 $j$ 收到消息后，更新 $V_j[k] = \max(V_j[k], V_i[k])$（对所有 $k$），且 $V_j[j] = V_j[j] + 1$。

## 3. 并发冲突判定与 Dynamo 实践
- 若 $V_A \le V_B$ 且不全相等，则事件 A 因果先于（Happened-before）事件 B。
- 若存在 $V_A[x] > V_B[x]$ 且 $V_A[y] < V_B[y]$，则 A 与 B 为并发事件（Concurrent），由应用层合并解决冲突（如购物车合并）。"""),

    ("gossip_protocol_cluster", "Gossip 流行病协议与去中心化拓扑", """# Gossip 流行病协议与去中心化拓扑

## 1. 核心原理
Gossip 是一种基于随机漫步、去中心化的点对点通信协议。
- 节点周期性地从已知存活列表中随机选择 $k$ 个邻居节点，发送自身持有的集群元数据或状态摘要。
- 通信复杂度低：在 $O(\log N)$ 周期内，信息可收敛覆盖到全局 $N$ 个节点。

## 2. 反熵 (Anti-Entropy) 与传言传播 (Rumor-Mongering)
- **传言传播**：新节点变更事件被当作“热点传言”高频随机扩散，直到接收到多数重复确认后降频。
- **反熵机制**：周期性比较两个节点的完整数据或 Merkle 树哈希值，修复网络瞬断造成的残留状态不一致。常用于 Cassandra、Redis Cluster 的节点存活探测。"""),

    ("consistent_hashing_virtual_nodes", "一致性哈希与虚拟节点平衡", """# 一致性哈希与虚拟节点平衡

## 1. 传统模数哈希的扩容死穴
传统分片采用 `hash(key) % N`。当增加或删除一个节点时，几乎所有数据对应的槽位都会发生改变，导致缓存雪崩或巨量数据迁移。

## 2. 哈希环与虚拟节点机制
- **哈希环 (0 到 $2^{32}-1$)**：将节点与数据键映射到同一连续环形空间，数据顺时针寻址定位到第一个节点。
- **虚拟节点 (Virtual Nodes)**：为解决物理节点分布不均产生的数据倾斜（Hot Spot），每个物理节点分配数百个虚拟节点（如 `node1#1`, `node1#2`），使哈希环上的槽位更加均匀平滑，节点变更时仅迁移 $1/N$ 的数据量。"""),

    ("distributed_locks_redlock", "分布式锁：Redis Redlock 算法与租约争议", """# 分布式锁：Redis Redlock 算法与租约争议

## 1. 基础单节点锁与死穴
单节点 Redis 分布式锁采用 `SET key value NX PX 30000`，配合随机 UUID 防止误删。但无法防范 Redis 宕机或主从异步复制导致的锁丢失。

## 2. Redlock 算法流程
在 $N$ 个独立的 Redis Master 上：
1. 客户端获取当前时间戳。
2. 依次向 $N$ 个节点请求加锁，设置远小于锁超时时间的网络等待。
3. 当且仅当在超过半数（Quorum）节点上成功获取锁，且总耗时小于锁有效时间时，锁才算获取成功。

## 3. Martin Kleppmann 的质疑与 GC 停顿
分布式学者 Martin 提出：GC 停顿（Stop-the-World）、网络阻塞或时钟跳变会导致客户端在不知情的情况下锁过期，而客户端随后执行的写入会与并发锁冲突。严格的互斥必须依靠**单调递增的栅栏令牌 (Fencing Token)**。"""),

    ("cap_pacelc_theorem", "CAP 定理与 PACELC 理论在生产架构中的抉择", """# CAP 定理与 PACELC 理论在生产架构中的抉择

## 1. CAP 定理重审
- **Consistency (强一致性)**：所有节点在同一时刻看到相同的数据。
- **Availability (高可用性)**：非故障节点必须对请求做出非错误响应。
- **Partition Tolerance (分区容忍性)**：网络断开时系统仍能工作。
由于分布式网络分区是物理必然（P 必须满足），系统只能在 CP（如 HBase, etcd）和 AP（如 Cassandra, CouchDB）中权衡。

## 2. PACELC 扩展理论
Daniel Abadi 指出 CAP 仅解释了分区（Partition）时的行为。PACELC 补全了常规无分区情况：
- **如果存在分区 (P)**：选择可用性 (A) 还是强一致性 (C)？
- **否则正常情况下 (Else)**：选择低延迟 (Latency, L) 还是强一致性 (Consistency, C)？
例如 MongoDB 通常属于 PC/EC（常态选延迟，分区选一致），而 DynamoDB 属于 PA/EL。"""),

    ("quorum_nwr_model", "Quorum NWR 调谐模型与可调节一致性", """# Quorum NWR 调谐模型与可调节一致性

## 1. 参数定义
- $N$：数据副本总数。
- $W$：写操作成功所必须确认的最小副本数。
- $R$：读操作成功所必须读取的最小副本数。

## 2. 鸽巢原理与一致性保证
- **强一致性 (Strong Consistency)**：当 $W + R > N$ 时，读集合与写集合必然存在至少一个重叠节点。结合版本号即可识别出最新提交的数据。
- **写优化配置**：$W=1, R=N$。写极快且高可用，但读性能严重下降。
- **读优化配置**：$W=N, R=1$。读操作只需访问任意单个节点即可，但写操作必须全员存活。
- **对称折中配置**：$W = \lfloor N/2 \rfloor + 1, R = \lfloor N/2 \rfloor + 1$，典型 Quorum 多数派。"""),

    ("byzantine_fault_tolerance_pbft", "拜占庭容错 PBFT 算法核心推导", """# 拜占庭容错 PBFT 算法核心推导

## 1. 拜占庭将军问题的容错边界
在存在恶意篡改、欺骗或软件 Bug（拜占庭错误）的分布式网络中，若要容忍 $f$ 个拜占庭故障节点，系统节点总数 $R$ 必须满足：
$$R \ge 3f + 1$$
因为在最坏情况下，有 $f$ 个恶意节点撒谎，$f$ 个无响应，$3f+1$ 保证了剩下的诚实节点能够以绝对多数达成共识。

## 2. 三阶段协议
1. **Pre-prepare**：主节点广播提案并分配视图编号与序列号。
2. **Prepare**：节点验证后广播 Prepare 消息，收集到 $2f$ 个匹配回复后进入 Prepared 状态。
3. **Commit**：节点广播 Commit 消息，收集到 $2f+1$ 个 Commit 消息后真正应用至状态机。"""),

    ("etcd_raft_mvcc", "etcd 的架构解密：Raft 共识与 bbolt MVCC 存储", """# etcd 的架构解密：Raft 共识与 bbolt MVCC 存储

## 1. 双层架构设计
etcd 分为两层：
- **共识层 (Raft)**：负责将写操作日志达成 Quorum 排序一致。
- **存储层 (MVCC + bbolt)**：内存中维护基于 B-Tree 的 `keyIndex`，磁盘底层采用基于 B+ Tree 的键值数据库 bbolt。

## 2. MVCC 修订版本号 (Revision)
- etcd 中每次事务修改，全局 `revision` 递增。
- 每个 key 的内部记录保留了 `create_revision`、`mod_revision` 和历史版本链。
- 这为 Kubernetes 提供了天然的增量变更监听（Watch）机制，客户端可以通过携带指定 revision 实现断点续传。"""),

    ("split_brain_fencing", "集群脑裂成因与防御：从 Quorum 到 STONITH", """# 集群脑裂成因与防御：从 Quorum 到 STONITH

## 1. 脑裂本质
网络分区导致集群被物理切割为多个无法互通的子集群，每个子集群误以为对方节点已挂，各自选举出新的主节点并接受并发写操作，导致元数据永久分叉损坏。

## 2. 防御策略
1. **奇数节点与多数派选举**：集群固定配置 3, 5, 7 节点，保证至多一个分区能凑齐 $n/2+1$ 的多数派。
2. **仲裁 Witness 节点**：在双机房部署中引入第三方仲裁节点打破僵局。
3. **STONITH (Shoot The Other Node In The Head)**：利用带外管理硬件（IPMI）直接切断失联节点的物理电源，实现绝对物理隔离。"""),

    ("distributed_tracing_dapper", "分布式链路追踪：Dapper 模型与 Trace/Span 机制", """# 分布式链路追踪：Dapper 模型与 Trace/Span 机制

## 1. 追踪数据模型
- **Trace**：代表一次端到端的分布式请求全生命周期，由唯一的 `TraceId` 标识。
- **Span**：代表链路中某一个服务或组件内的一次独立工作单元，由 `SpanId` 标识。
- **ParentSpanId**：标识父子因果关系，多个 Span 构成一棵有向无环树（DAG）。

## 2. 上下文传播与采样率
- **B3 / W3C TraceContext**：通过 HTTP Header 或 gRPC Metadata 随网络调用透明透传追踪 ID。
- **自适应采样 (Adaptive Sampling)**：为了压制海量请求对存储和网络带宽的消耗，系统通常采用基于哈希的固定比例采样或尾部异常自适应采样。"""),

    ("circuit_breaker_bulkhead", "微服务稳定性底座：断路器与舱壁隔离模式", """# 微服务稳定性底座：断路器与舱壁隔离模式

## 1. 断路器三态模型 (Circuit Breaker)
- **Closed (闭合)**：正常流量放行，统计滑动时间窗口内的失败率或慢调用比例。
- **Open (熔断开路)**：失败率超阈值，断路器跳闸，所有后续请求直接快速失败（Fast-Fail）或走降级回调。
- **Half-Open (半开)**：经过冷却时间后，放行少量探测流量。若全部成功则恢复为 Closed，若失败则退回 Open。

## 2. 舱壁隔离 (Bulkhead Pattern)
仿照轮船水密隔舱设计，将不同下游服务的调用线程池或信号量相互物理隔离。防止某单一外部系统延迟暴增耗尽全局主线程池，避免雪崩效应。"""),

    ("kubernetes_scheduler_architecture", "Kubernetes Kube-scheduler 调度器内核机制", """# Kubernetes Kube-scheduler 调度器内核机制

## 1. 调度流水线两大阶段
- **过滤阶段 (Filtering / Predicates)**：找出满足 Pod 资源和拓扑约束的可用 Node 候选集。
  - 检查项：PodFitsResources、NodeName、PodFitsHostPorts、NodeAffinity、Toleration 与 Taint 匹配。
- **打分阶段 (Scoring / Priorities)**：对过滤后的 Node 进行综合打分（0-100分）。
  - 打分维度：NodeResourcesBalancedAllocation（CPU与内存均衡度）、ImageLocalityPriority（镜像本地缓存度）。

## 2. 调度上下文与并发乐观绑定
调度器在内存中保留一份 NodeCache。调度决策在本地内存缓存中执行快速的“乐观预占（Assume Pod）”，随后异步将 Binding 对象写入 API Server，从而避免长时间锁占用。"""),

    ("grpc_http2_multiplexing", "gRPC 架构与 HTTP/2 多路复用原理", """# gRPC 架构与 HTTP/2 多路复用原理

## 1. 二进制分帧层 (Binary Framing)
HTTP/2 抛弃了 HTTP/1.1 的纯文本格式，将数据切分为帧（Frames）：HEADERS 帧、DATA 帧、SETTINGS 帧等。

## 2. 多路复用 (Multiplexing) 解决队头阻塞
- 单个 TCP 连接上可以并发交替传输成百上千个独立的流（Streams）。
- 每个流有唯一的 Stream ID，彻底消除了 HTTP/1.1 应用层的队头阻塞（Head-of-Line Blocking）。
- 结合 Protobuf 强类型二进制序列化，gRPC 在高并发微服务场景下具备远超 REST/JSON 的编解码效率与连接复用率。"""),

    ("event_sourcing_cqrs", "事件溯源 (Event Sourcing) 与 CQRS 架构解耦", """# 事件溯源 (Event Sourcing) 与 CQRS 架构解耦

## 1. 事件溯源 (Event Sourcing)
- 放弃传统关系型数据库将“最终状态”直接覆写旧数据的做法。
- 将系统的一切状态变更记录为不可变的原子事件流（Append-Only Event Store）。
- 业务当前状态是历史所有事件从初始状态按序重放（Replay）推导出来的投影（Projection）。

## 2. CQRS 命令查询职责分离
- **Command 侧**：负责处理写请求、校验业务约束并写入事件存储，强调强一致性。
- **Query 侧**：订阅事件总线，异步将事件增量物化为针对特定查询优化的只读视图（如 ElasticSearch、Redis），提供超高并发读能力。"""),

    ("idempotency_token_bucket", "分布式接口幂等性设计与令牌桶限流", """# 分布式接口幂等性设计与令牌桶限流

## 1. 幂等性设计模式
- **唯一业务键约束**：数据库建立唯一索引（Unique Constraint）。
- **分布式去重表**：利用 Redis `SETNX` 写入防重令牌（Token），设定合理 TTL。
- **状态机悲观幂等**：基于行级版本号的 CAS 更新（如 `UPDATE orders SET status=PAID WHERE id=1 AND status=UNPAID`）。

## 2. 令牌桶限流算法 (Token Bucket)
- 算法以恒定速率 $r$ 向容量为 $b$ 的桶内注入令牌。
- 请求到达时需消耗指定数量令牌；桶空则限流排队或拒绝。
- **对比漏桶 (Leaky Bucket)**：令牌桶允许在突发流量时一次性消耗桶内积攒的全部令牌，具备极佳的应对突发突刺流量的能力。"""),

    ("distributed_cache_consistency", "分布式缓存与数据库双写一致性机制", """# 分布式缓存与数据库双写一致性机制

## 1. 经典更新策略陷阱
- **先删缓存后更库**：并发读请求在更新数据库完成前读到旧数据并重新回填缓存，导致脏数据永久存在。
- **先更库后更缓存**：两个并发写请求可能因网络延迟出现写乱序，后写的数据先覆盖缓存。

## 2. Cache-Aside 最佳实践
- **读流程**：先读缓存，命中则返回；未命中则查数据库并写入缓存。
- **写流程**：**先更新数据库，再删除缓存**。虽然存在理论上极低概率的读写交织脏数据，但结合缓存过期时间（TTL），能保证绝大多数业务场景下的最终一致性。
- **最终兜底方案**：通过解析数据库 binlog（如 Canal）异步延时删除缓存（Cache-Aside + Binlog Async Invalidation）。""")
]
