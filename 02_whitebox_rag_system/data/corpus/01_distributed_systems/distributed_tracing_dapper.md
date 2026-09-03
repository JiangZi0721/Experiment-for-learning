# 分布式链路追踪：Dapper 模型与 Trace/Span 机制

## 1. 追踪数据模型
- **Trace**：代表一次端到端的分布式请求全生命周期，由唯一的 `TraceId` 标识。
- **Span**：代表链路中某一个服务或组件内的一次独立工作单元，由 `SpanId` 标识。
- **ParentSpanId**：标识父子因果关系，多个 Span 构成一棵有向无环树（DAG）。

## 2. 上下文传播与采样率
- **B3 / W3C TraceContext**：通过 HTTP Header 或 gRPC Metadata 随网络调用透明透传追踪 ID。
- **自适应采样 (Adaptive Sampling)**：为了压制海量请求对存储和网络带宽的消耗，系统通常采用基于哈希的固定比例采样或尾部异常自适应采样。
