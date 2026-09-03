# 大模型连续批处理 (Continuous Batching / Dynamic Batching)

## 1. 静态批处理 (Static Batching) 的短板
传统推理由客户端批量提交 $N$ 个请求。系统必须等待**最长的一个请求完全生成完毕**才能释放该批次，导致短序列请求白白等待（GPU 利用率出现严重尾部空洞）。

## 2. 迭代级调度 (Iteration-level Scheduling)
- Orca 与 vLLM 引入了连续批处理：调度器在每一个生成步骤（Step）结束后介入。
- 一旦某个请求生成结束符 `<|endoftext|>`，系统立即将其弹出并释放显存；同时将队列中新到达的请求动态插入当前 Step 的空位继续并发计算，系统吞吐量提升 2~4 倍。
