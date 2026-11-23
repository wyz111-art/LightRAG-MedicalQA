"""Build the local medical KG from the Huatuo26 sample (batch + checkpoint + resume)."""
import asyncio
import json
import os
import random
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import ollama

from lightrag import LightRAG
from lightrag.utils import wrap_embedding_func_with_attrs

DATA_FILE = Path(os.getenv("MEDICAL_DATA_FILE", "data/sample_6000.jsonl"))
WORKING_DIR = Path(os.getenv("MEDICAL_WORKING_DIR", "medical_kg"))
LIMIT = int(os.getenv("MEDICAL_DOC_LIMIT", "6000"))
BATCH_SIZE = int(os.getenv("MEDICAL_INSERT_BATCH", "100"))
EMBED_BATCH_SIZE = int(os.getenv("MEDICAL_EMBED_BATCH", "32"))
LLM_MODEL = os.getenv("MEDICAL_LLM_MODEL", "qwen2.5:14b")
EMBED_MODEL = os.getenv("MEDICAL_EMBED_MODEL", "nomic-embed-text")
OLLAMA_HOST = os.getenv("OLLAMA_HOST")
client = ollama.AsyncClient(host=OLLAMA_HOST) if OLLAMA_HOST else ollama.AsyncClient()

NS = {"g": "http://graphml.graphdrawing.org/xmlns"}


async def llm_model_func(prompt, system_prompt=None, **kwargs):
    response = await client.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt or ""},
            {"role": "user", "content": prompt},
        ],
    )
    return response["message"]["content"]


@wrap_embedding_func_with_attrs(
    embedding_dim=768, max_token_size=8192, model_name=EMBED_MODEL
)
async def embedding_func(texts, embedding_dim=None):
    vectors = []
    for start in range(0, len(texts), EMBED_BATCH_SIZE):
        response = await client.embed(
            model=EMBED_MODEL, input=texts[start : start + EMBED_BATCH_SIZE]
        )
        vectors.extend(response["embeddings"])
    return np.asarray(vectors, dtype="float32")


def load_documents():
    docs = []
    with DATA_FILE.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {DATA_FILE}:{line_number}") from exc
            docs.append(
                "\n".join(
                    (
                        f"科室：{row.get('label', '')}",
                        f"相关疾病：{row.get('related_diseases', '')}",
                        f"问题：{row.get('question', '')}",
                        f"回答：{row.get('answer', '')}",
                    )
                )
            )

    random.seed(42)
    random.shuffle(docs)

    if LIMIT > 0:
        docs = docs[:LIMIT]

    print(f"Loaded {len(docs)} documents from {DATA_FILE}")
    return docs


def graph_stats():
    grf = WORKING_DIR / "graph_chunk_entity_relation.graphml"
    if not grf.is_file():
        return 0, 0
    root = ET.parse(grf).getroot()
    return len(root.findall(".//g:node", NS)), len(root.findall(".//g:edge", NS))


def get_resume_offset():
    """从 doc_status 读取已处理文档数，作为下次起点"""
    status_file = WORKING_DIR / "kv_store_doc_status.json"
    if not status_file.is_file():
        return 0
    status = json.load(open(status_file, encoding="utf-8"))
    processed = sum(1 for v in status.values() if v.get("status") == "processed")
    return processed


async def main():
    if BATCH_SIZE < 1 or EMBED_BATCH_SIZE < 1:
        raise ValueError("batch limits must be positive")
    if not DATA_FILE.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    docs = load_documents()
    if not docs:
        raise ValueError(f"No documents found in {DATA_FILE}")

    offset = get_resume_offset()
    if offset > 0:
        print(f"\n=== 断点续跑：已处理 {offset} 条，从第 {offset + 1} 条开始 ===")
        docs = docs[offset:]

    print(f"Ollama models: LLM={LLM_MODEL}, embeddings={EMBED_MODEL}")
    print(f"Working dir: {WORKING_DIR.resolve()}")

    rag = LightRAG(
        working_dir=str(WORKING_DIR),
        llm_model_func=llm_model_func,
        embedding_func=embedding_func,
        max_parallel_insert=2,
        llm_model_max_async=2,
        embedding_func_max_async=4,
        embedding_batch_num=EMBED_BATCH_SIZE,
        entity_extract_max_gleaning=0,
        chunk_token_size=800,
        chunk_overlap_token_size=50,
        addon_params={"language": "Chinese"},
    )

    await rag.initialize_storages()

    started = time.perf_counter()
    processed = 0
    try:
        for start in range(0, len(docs), BATCH_SIZE):
            batch = docs[start : start + BATCH_SIZE]
            await rag.ainsert(batch)
            processed += len(batch)
            elapsed = time.perf_counter() - started
            rate = processed / elapsed if elapsed else 0

            n_nodes, n_edges = graph_stats()
            print(
                f"\n=== 批次完成: {processed}/{len(docs)} ==="
                f"\n当前图谱: {n_nodes} 节点, {n_edges} 边"
                f"\n耗时 {elapsed/60:.1f} 分钟"
            )

            if start + BATCH_SIZE < len(docs):
                print(
                    f"\n--- 检查图谱：打开 {WORKING_DIR / 'edges_view.html'} "
                    f"或 {WORKING_DIR / 'graph_view.html'} 看一下 ---"
                )
                print("--- 按回车继续下一批，Ctrl+C 停止 ---")
                input()
    finally:
        await rag.finalize_storages()

    elapsed = time.perf_counter() - started
    print(f"\nBuild complete: {processed} documents in {elapsed/60:.1f} min")
    print(f"Storage: {WORKING_DIR.resolve()}")


if __name__ == "__main__":
    asyncio.run(main())
