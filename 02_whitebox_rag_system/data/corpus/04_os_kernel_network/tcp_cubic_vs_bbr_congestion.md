# TCP 拥塞控制演进：基于丢包的 CUBIC vs 基于吞吐瓶颈的 BBR

## 1. 传统 CUBIC (基于丢包的被动控制)
- 认为“丢包即拥塞”。将拥塞窗口（cwnd）按照三次函数增长，直到发生网络丢包才急剧减半窗口。
- **缓冲区膨胀 (Bufferbloat) 死穴**：在当今中间路由器拥有海量缓冲队列的环境下，CUBIC 会填满队列才丢包，导致网络往返时延（RTT）急剧恶化。

## 2. Google BBR (Bottleneck Bandwidth and RTT)
- 抛弃丢包信号，采用主动物理建模。
- **实时探测两个物理极值**：网络最大可用带宽（Max BtlBw）与最小传播往返时间（Min RTprop）。
- 将在网数据包量（In-Flight Data）严格控制在 $BDP = \text{BtlBw} \times \text{RTprop}$，既打满链路带宽又绝不填塞路由器队列，在弱网或跨洋长延迟链路中吞吐量可提升数倍至数十倍。
