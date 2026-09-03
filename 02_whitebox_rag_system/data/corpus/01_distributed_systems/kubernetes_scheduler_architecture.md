# Kubernetes Kube-scheduler 调度器内核机制

## 1. 调度流水线两大阶段
- **过滤阶段 (Filtering / Predicates)**：找出满足 Pod 资源和拓扑约束的可用 Node 候选集。
  - 检查项：PodFitsResources、NodeName、PodFitsHostPorts、NodeAffinity、Toleration 与 Taint 匹配。
- **打分阶段 (Scoring / Priorities)**：对过滤后的 Node 进行综合打分（0-100分）。
  - 打分维度：NodeResourcesBalancedAllocation（CPU与内存均衡度）、ImageLocalityPriority（镜像本地缓存度）。

## 2. 调度上下文与并发乐观绑定
调度器在内存中保留一份 NodeCache。调度决策在本地内存缓存中执行快速的“乐观预占（Assume Pod）”，随后异步将 Binding 对象写入 API Server，从而避免长时间锁占用。
