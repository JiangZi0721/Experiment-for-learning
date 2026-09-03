# TCP Fast Open (TFO): 消除三次握手 1-RTT 延迟

## 1. 传统握手的 1-RTT 延迟惩罚
客户端在发出 SYN 后的整整一个往返时间（1-RTT）内不能发送任何实际的 HTTP Payload，在短连接移动网络中极大拉长首包时间。

## 2. TFO 的加密 Cookie 验证机制
- **首次连接请求**：Client 发送带 TFO 选项的 SYN，Server 验证后在 SYN-ACK 中颁发一个加密的 TFO Cookie。
- **后续连接极速传输**：Client 再次连接时，直接在 **SYN 报文中携带该 TFO Cookie 和真实的 HTTP 业务数据**。
- Server 验证 Cookie 合法后，立即将数据送交应用层，在握手尚未完全完成之前即可开始处理业务逻辑，实现真正的 0-RTT 首包加速。
