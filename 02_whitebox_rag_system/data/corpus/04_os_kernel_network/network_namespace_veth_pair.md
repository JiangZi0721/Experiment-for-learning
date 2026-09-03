# 容器网络底层基石：Network Namespace 与 veth-pair 管道

## 1. Network Namespace 逻辑隔离
Linux 内核通过 Namespace 实现轻量级容器隔离。独立的 Net Namespace 拥有各自独立的路由表、iptables 规则链、网络设备列表与 Socket 端口空间。

## 2. veth-pair 虚拟以太网对
- 像一根双向连通的“虚拟网线”，必须成对创建。
- 从一端发出的数据包会被内核无缝重定向从另一端直接接收。
- **容器互联模式**：veth 的一端放置在容器内部重命名为 `eth0`；另一端插在宿主机的虚拟网桥（如 `docker0` 或 `cni0`）上，配合网桥广播与路由转发实现跨容器通信。
