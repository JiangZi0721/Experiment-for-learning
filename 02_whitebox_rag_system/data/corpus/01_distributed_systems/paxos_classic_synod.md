# Classic Paxos 决议推演与两阶段提交

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
- Acceptor 收到 Accept 请求时，只要它此前没有承诺过只接受大于 $n$ 的提案，就必须批准该提案。
