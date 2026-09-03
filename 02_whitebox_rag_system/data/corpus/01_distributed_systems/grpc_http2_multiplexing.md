# gRPC 架构与 HTTP/2 多路复用原理

## 1. 二进制分帧层 (Binary Framing)
HTTP/2 抛弃了 HTTP/1.1 的纯文本格式，将数据切分为帧（Frames）：HEADERS 帧、DATA 帧、SETTINGS 帧等。

## 2. 多路复用 (Multiplexing) 解决队头阻塞
- 单个 TCP 连接上可以并发交替传输成百上千个独立的流（Streams）。
- 每个流有唯一的 Stream ID，彻底消除了 HTTP/1.1 应用层的队头阻塞（Head-of-Line Blocking）。
- 结合 Protobuf 强类型二进制序列化，gRPC 在高并发微服务场景下具备远超 REST/JSON 的编解码效率与连接复用率。
