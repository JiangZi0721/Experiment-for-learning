# TCP 三次握手状态迁移与 SYN Flood 洪水攻击防御

## 1. 三次握手状态机
1. Client 发送 `SYN` (seq=x)，进入 `SYN_SENT`。
2. Server 回复 `SYN+ACK` (seq=y, ack=x+1)，进入 `SYN_RCVD`，将连接放入**半连接队列 (SYN Queue)**。
3. Client 回复 `ACK` (seq=x+1, ack=y+1)，进入 `ESTABLISHED`；Server 收到后将连接移入**全连接队列 (Accept Queue)**。

## 2. SYN Flood 攻击本质
黑客伪造大量虚假源 IP 发送大量 SYN 请求，且故意不回复最后的 ACK。导致 Server 的半连接队列迅速打满，正常合法用户的连接请求全部被丢弃。

## 3. SYN Cookie 防御机制
- 当半连接队列满时，Server 不分配 `struct request_sock` 结构。
- 依据时间戳、源/目的 IP、端口以及安全密钥通过哈希计算出初始序列号 $seq_y$（即 SYN Cookie）。
- 只有当合法的第三次 ACK 到达且其中的 $ack$ 能被成功还原验证时，才分配连接资源，彻底免受队列占满限制。
