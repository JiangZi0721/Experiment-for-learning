# 分块预填充 (Chunked Prefill) 与 Prompt Caching 系统

## 1. Prefill 阶段与 Decode 阶段的算力冲突
- **Prefill (首字生成)**：Compute-bound，计算密度极高，耗时长，会瞬间卡死正在执行的轻量 Decode 请求，造成 TTFT（首字延迟）急剧劣化。
- **Decode (增量流式)**：Memory-bandwidth-bound。

## 2. Chunked Prefill 分块技术
- 将一个几千 Token 的超长 Prompt 切分成多个固定大小的 Chunk（如 512）。
- 将一个 Prefill Chunk 与若干处于 Decode 阶段的请求打包在同一个 Batch 内并发执行，彻底熨平算力波动。

## 3. Prompt Caching (上下文提示词缓存)
对于多轮对话中重复出现的高频 System Prompt 或前序知识库，显存中保留已计算好的 KV 块哈希索引，新请求直接复用缓存，跳过 Prefill 计算。
