# Socket 缓冲区内核调优与 TCP 窗口缩放因子 (Window Scale)

## 1. 带宽时延积 (BDP) 与窗口瓶颈
$$BDP = \text{Bandwidth} \times \text{RTT}$$
为了使网络管道始终填满数据，发送端与接收端的滑动窗口大小必须至少等于 BDP。

## 2. 传统 16-bit 窗口限制与 Window Scale 扩展
TCP 报文头中原始的 `Window Size` 字段仅有 16 位，最大只能表示 64KB，在千兆万兆网络中极速变成瓶颈。
- **Window Scale (RFC 1323)**：在握手 SYN 阶段协商缩放移位因子（0~14）。实际窗口大小为 $\text{Window} \times 2^{scale}$，最大可扩展到 1GB。

## 3. 内核自动调谐 (Autotuning)
Linux 针对 `rmem` 和 `wmem` 提供了 `[min, default, max]` 三元组，内核根据当前连接的实时 RTT 和传输速率自动弹性调整 Socket 缓冲区。
