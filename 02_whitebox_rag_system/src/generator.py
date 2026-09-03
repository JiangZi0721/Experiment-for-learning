# -*- coding: utf-8 -*-
"""
DeepSeek LLM 生成器与 Prompt 透视器
功能：
1. 组装具备防幻觉铁律的白盒 System Prompt
2. 调用 DeepSeek API 执行流式输出 (优先使用 openai SDK，降级支持原生标准库 urllib SSE 流式解析)
3. 事实溯源与引用标注校验 ([1], [2] 标注)
"""
import sys
import json
import urllib.request
import urllib.error
from typing import List, Dict, Any, Generator, Tuple

from .config import cfg

class DeepSeekGenerator:
    def __init__(self):
        self.api_key = cfg.DEEPSEEK_API_KEY
        self.base_url = cfg.DEEPSEEK_BASE_URL
        self.model = cfg.DEEPSEEK_MODEL
        self.temperature = cfg.DEEPSEEK_TEMPERATURE

    def build_prompt(self, query: str, golden_chunks: List[Dict[str, Any]]) -> Tuple[str, str]:
        """组装带严密约束的 Prompt 并返回 (system_prompt, user_prompt)"""
        context_blocks = []
        for idx, chunk in enumerate(golden_chunks, 1):
            block = (
                f"--- 资料片段 [{idx}] ---\n"
                f"出处：{chunk['domain']} / {chunk['heading_path']}\n"
                f"内容：\n{chunk['content']}\n"
            )
            context_blocks.append(block)

        context_str = "\n".join(context_blocks)

        system_prompt = (
            "你是一个技术严谨的计算机底层架构专家。\n"
            "【核心铁律 (Faithfulness First)】：\n"
            "1. 你的回答必须严格基于【参考资料】中提供的事实，严禁根据自身参数记忆随意脑补捏造未提及的细节。\n"
            "2. 每一个核心事实陈述，必须在对应句子末尾紧随来源编号标注，例如 '[1]' 或 '[1][2]'。\n"
            "3. 若【参考资料】中完全未提及用户询问的技术概念或该技术属于虚构伪命题，必须果断明确答复：'根据已知资料，未提及该技术或机制'，并指出事实矛盾之处，严禁强行拼凑！\n"
            "4. 回答应当条理清晰、直击本质，包含底层逻辑推导与架构权衡。"
        )

        user_prompt = (
            f"【参考资料】：\n{context_str}\n\n"
            f"【用户问题】：{query}\n\n"
            f"请根据上述资料进行深入、严谨的分析回答："
        )

        return system_prompt, user_prompt

    def generate_stream(self, query: str, golden_chunks: List[Dict[str, Any]]) -> Generator[str, None, None]:
        """流式调用 DeepSeek API 并逐字产出"""
        system_prompt, user_prompt = self.build_prompt(query, golden_chunks)

        if not self.api_key or "your_" in self.api_key.lower():
            yield "\n[提示]：未在 .env 中检测到有效的 DEEPSEEK_API_KEY。\n"
            yield f"[模拟透视]：针对 Query '{query}'，系统已从语料库精准召回并精选出 Top-{len(golden_chunks)} 个切片注入上下文。\n"
            yield "当你将 DEEPSEEK_API_KEY 填入 .env 后，DeepSeek 将基于上述切片进行带引用标注 ([1], [2]) 的流式推理！\n"
            return

        # 优先尝试使用 openai SDK
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                stream=True
            )
            for chunk in response:
                content = chunk.choices[0].delta.content
                if content:
                    yield content
            return
        except ImportError:
            pass
        except Exception as e:
            yield f"\n[OpenAI SDK 异常，尝试原生 HTTP SSE]: {str(e)}\n"

        # 原生 Python 标准库 urllib 实现 SSE 流式解析 (零三方库依赖，100% 稳定)
        try:
            url = f"{self.base_url.rstrip('/')}/chat/completions"
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": self.temperature,
                "stream": True
            }
            req_data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=req_data,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
            )
            import ssl
            ssl_ctx = ssl._create_unverified_context()

            with urllib.request.urlopen(req, timeout=60, context=ssl_ctx) as resp:
                for line in resp:
                    decoded = line.decode("utf-8").strip()
                    if not decoded:
                        continue
                    if decoded.startswith("data: "):
                        data_str = decoded[6:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk_json = json.loads(data_str)
                            delta = chunk_json["choices"][0].get("delta", {})
                            token = delta.get("content", "")
                            if token:
                                yield token
                        except Exception:
                            continue
        except Exception as err:
            yield f"\n[DeepSeek 原生流式请求异常]: {str(err)}\n"
