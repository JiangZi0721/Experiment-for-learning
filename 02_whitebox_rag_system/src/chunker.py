# -*- coding: utf-8 -*-
"""
AST 结构化 Markdown 切分器 (Structural Markdown Chunker)
遵循设计哲学：拒绝暴力滑动窗口截断，基于 Markdown 标题树切块，
并在每个切片头部注入父子标题层级面包屑 (Breadcrumbs)，彻底杜绝主谓宾丢失与指代不清。
"""
import re
from pathlib import Path
from typing import List, Dict, Any

class MarkdownChunk:
    def __init__(self, chunk_id: str, doc_id: str, domain: str, heading_path: str, content: str):
        self.chunk_id = chunk_id
        self.doc_id = doc_id
        self.domain = domain
        self.heading_path = heading_path
        self.raw_content = content.strip()
        # 组装携带面包屑元数据的可检索文本
        self.text_for_retrieval = f"【主题路径：{domain} > {heading_path}】\n{self.raw_content}"
        self.char_count = len(self.text_for_retrieval)
        # 预估 Token 数量（中文通常 1.5 字符约为 1 Token，英文以空格/子词计）
        self.token_estimate = int(len(self.text_for_retrieval) * 0.7)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "domain": self.domain,
            "heading_path": self.heading_path,
            "content": self.text_for_retrieval,
            "char_count": self.char_count,
            "token_estimate": self.token_estimate
        }

class StructuralMarkdownChunker:
    def __init__(self, min_chunk_chars: int = 50, max_chunk_chars: int = 1200):
        self.min_chunk_chars = min_chunk_chars
        self.max_chunk_chars = max_chunk_chars

    def chunk_file(self, file_path: Path, domain: str) -> List[MarkdownChunk]:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        doc_id = file_path.stem
        chunks = []
        
        # 状态机：跟踪标题层级栈
        heading_stack = []  # [(level, text)]
        current_chunk_lines = []
        chunk_idx = 1

        def get_current_heading_path():
            if not heading_stack:
                return doc_id
            return " > ".join([h[1] for h in heading_stack])

        for line in lines:
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line.strip())
            if header_match:
                level = len(header_match.group(1))
                title = header_match.group(2).strip()

                # 如果当前累积的段落内容足够，则先封装为一个切片
                text_block = "".join(current_chunk_lines).strip()
                if len(text_block) >= self.min_chunk_chars:
                    c_id = f"{domain[:4]}_{doc_id}#{chunk_idx:02d}"
                    chunks.append(MarkdownChunk(
                        chunk_id=c_id,
                        doc_id=doc_id,
                        domain=domain,
                        heading_path=get_current_heading_path(),
                        content=text_block
                    ))
                    chunk_idx += 1
                    current_chunk_lines = []

                # 更新标题栈
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                heading_stack.append((level, title))
            else:
                current_chunk_lines.append(line)

        # 处理末尾剩余的段落
        text_block = "".join(current_chunk_lines).strip()
        if text_block:
            c_id = f"{domain[:4]}_{doc_id}#{chunk_idx:02d}"
            chunks.append(MarkdownChunk(
                chunk_id=c_id,
                doc_id=doc_id,
                domain=domain,
                heading_path=get_current_heading_path(),
                content=text_block
            ))

        return chunks

    def chunk_corpus(self, corpus_root: Path) -> List[MarkdownChunk]:
        """批量遍历全量语料库生成 Chunks"""
        all_chunks = []
        for domain_dir in sorted(corpus_root.iterdir()):
            if domain_dir.is_dir():
                domain_name = domain_dir.name
                for md_file in sorted(domain_dir.glob("*.md")):
                    chunks = self.chunk_file(md_file, domain_name)
                    all_chunks.extend(chunks)
        return all_chunks
