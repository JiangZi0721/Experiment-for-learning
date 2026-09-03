# TCP 四次挥手 TIME_WAIT 状态机与 SO_REUSEADDR 机制

## 1. 为何必须维持 2MSL 的 TIME_WAIT 状态
主动关闭连接的一方在收到对端 FIN 并回复 ACK 后，必须在此状态等待 $2\times \text{MSL}$（最大报文生存时间，Linux 默认 60s）：
- **可靠终止连接**：防止最后发出的 ACK 报文丢失，确保如果对端因超时重发 FIN，本端仍能回复 ACK。
- **消除迷走报文干扰**：保证本连接在网络中延迟残留的所有历史旧报文全部自然消亡，不至于污染后续复用相同四元组的新连接。

## 2. SO_REUSEADDR 与端口耗尽
高并发短连接服务器下，大量 Socket 卡在 TIME_WAIT 会耗尽可用端口。通过设置 `SO_REUSEADDR`，允许处于 TIME_WAIT 状态的端口被新创建的 Socket 立即绑定复用。
