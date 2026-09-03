# TCP 半连接队列与全连接队列溢出诊断

## 1. 队列满载的物理表现
- **半连接队列溢出**：收到 SYN 时若队列满且未开启 SYN Cookie，Server 直接丢弃 SYN，Client 超时重传。
- **全连接队列溢出**：三次握手完成，但上层应用（如 Java, Nginx）由于 CPU 繁忙未及时执行 `accept()`。
  - 由 `/proc/sys/net/ipv4/tcp_abort_on_overflow` 控制：若为 0，Server 直接丢弃最后的 ACK，假装未收到，迫使 Client 重发；若为 1，直接向 Client 发送 RST 强制重置连接。

## 2. 诊断命令
- `ss -lnt` 中的 `Send-Q` 代表全连接队列的最大容量（`backlog` 参数），`Recv-Q` 代表当前等待 `accept()` 的连接数。
- 当 `Recv-Q > Send-Q` 时，意味着全连接队列已彻底被打满，必须紧急调大参数或优化应用层消费速度。
