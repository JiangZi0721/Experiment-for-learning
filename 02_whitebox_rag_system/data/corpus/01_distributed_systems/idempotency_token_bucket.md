# 分布式接口幂等性设计与令牌桶限流

## 1. 幂等性设计模式
- **唯一业务键约束**：数据库建立唯一索引（Unique Constraint）。
- **分布式去重表**：利用 Redis `SETNX` 写入防重令牌（Token），设定合理 TTL。
- **状态机悲观幂等**：基于行级版本号的 CAS 更新（如 `UPDATE orders SET status=PAID WHERE id=1 AND status=UNPAID`）。

## 2. 令牌桶限流算法 (Token Bucket)
- 算法以恒定速率 $r$ 向容量为 $b$ 的桶内注入令牌。
- 请求到达时需消耗指定数量令牌；桶空则限流排队或拒绝。
- **对比漏桶 (Leaky Bucket)**：令牌桶允许在突发流量时一次性消耗桶内积攒的全部令牌，具备极佳的应对突发突刺流量的能力。
