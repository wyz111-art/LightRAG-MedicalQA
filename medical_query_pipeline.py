"""Route medical questions to graph, text, or hybrid LightRAG retrieval."""

import argparse
import asyncio
import json
import os
from typing import Any

import numpy as np
import ollama

from lightrag import LightRAG, QueryParam
from lightrag.utils import wrap_embedding_func_with_attrs
from medical_graph_query import MedicalGraphRetriever
from medical_intent import MedicalIntentRecognizer


WORKING_DIR = os.getenv("MEDICAL_WORKING_DIR", "medical_kg")
LLM_MODEL = os.getenv("MEDICAL_LLM_MODEL", "qwen2.5:14b")
EMBED_MODEL = os.getenv("MEDICAL_EMBED_MODEL", "nomic-embed-text")
EMBED_BATCH_SIZE = int(os.getenv("MEDICAL_EMBED_BATCH", "32"))
OLLAMA_HOST = os.getenv("OLLAMA_HOST")


def _create_ollama_client() -> ollama.AsyncClient:
    return ollama.AsyncClient(host=OLLAMA_HOST) if OLLAMA_HOST else ollama.AsyncClient()


def _build_llm_model_func(client: ollama.AsyncClient):
    async def llm_model_func(prompt, system_prompt=None, **kwargs):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        for message in kwargs.get("history_messages") or []:
            if (
                isinstance(message, dict)
                and message.get("role")
                and message.get("content")
            ):
                messages.append(
                    {"role": message["role"], "content": str(message["content"])}
                )
        messages.append({"role": "user", "content": prompt})
        response = await client.chat(model=LLM_MODEL, messages=messages)
        return response["message"]["content"]

    return llm_model_func


def _build_embedding_func(client: ollama.AsyncClient):
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

    return embedding_func


class MedicalQueryPipeline:
    def __init__(self, working_dir: str = WORKING_DIR):
        self.working_dir = working_dir
        self.intent_recognizer = MedicalIntentRecognizer()
        self.graph_retriever = MedicalGraphRetriever(working_dir)
        self._ollama_client = _create_ollama_client()
        self._llm_model_func = _build_llm_model_func(self._ollama_client)
        self._embedding_func = _build_embedding_func(self._ollama_client)
        self._rag: LightRAG | None = None

    async def _ensure_rag(self) -> LightRAG:
        if self._rag is None:
            self._rag = LightRAG(
                working_dir=self.working_dir,
                llm_model_func=self._llm_model_func,
                embedding_func=self._embedding_func,
            )
            await self._rag.initialize_storages()
        return self._rag

    async def initialize(self) -> None:
        """Preload LightRAG storages for a long-lived query service."""
        await self._ensure_rag()

    async def close(self) -> None:
        try:
            if self._rag is not None:
                await self._rag.finalize_storages()
                self._rag = None
        finally:
            try:
                await self.intent_recognizer.close()
            finally:
                await self._ollama_client.close()

    async def _lightrag_retrieval(
        self, question: str, mode: str
    ) -> dict[str, Any]:
        rag = await self._ensure_rag()
        query_mode = "naive" if mode == "text" else "hybrid"
        result = await rag.aquery_data(
            question,
            param=QueryParam(
                mode=query_mode,
                top_k=20,
                chunk_top_k=10,
                enable_rerank=False,
            ),
        )
        data = result.get("data") or {}
        return {
            "mode": query_mode,
            "status": result.get("status"),
            "message": result.get("message"),
            "entities": data.get("entities") or [],
            "relationships": data.get("relationships") or [],
            "chunks": data.get("chunks") or [],
            "references": data.get("references") or [],
            "metadata": result.get("metadata") or {},
        }

    async def retrieve(self, question: str) -> dict[str, Any]:
        question = question.strip()
        if not question:
            raise ValueError("question cannot be empty")

        intent = await self.intent_recognizer.recognize(question)
        intent_data = intent.to_dict()
        graph_queries = []

        if intent.relation:
            for entity in intent.entities:
                graph_result = self.graph_retriever.query_relation(
                    entity["name"],
                    intent.relation,
                    limit=20,
                    evidence_per_relation=2,
                )
                graph_queries.append(graph_result)

        if intent.retrieval_mode == "graph" and intent.relation:
            graph_match_count = sum(
                item.get("result_count", 0) for item in graph_queries
            )
            if graph_match_count:
                return {
                    "question": question,
                    "route_requested": "graph",
                    "route_used": "graph",
                    "fallback_reason": None,
                    "intent": intent_data,
                    "graph_queries": graph_queries,
                    "text_retrieval": None,
                }

            fallback_reason = (
                "no_entities_extracted"
                if not intent.entities
                else "no_graph_relation_matches"
            )
            text_retrieval = await self._lightrag_retrieval(question, "text")
            return {
                "question": question,
                "route_requested": "graph",
                "route_used": "text_fallback",
                "fallback_reason": fallback_reason,
                "intent": intent_data,
                "graph_queries": graph_queries,
                "text_retrieval": text_retrieval,
            }

        text_retrieval = await self._lightrag_retrieval(
            question, intent.retrieval_mode
        )
        return {
            "question": question,
            "route_requested": intent.retrieval_mode,
            "route_used": intent.retrieval_mode,
            "fallback_reason": None,
            "intent": intent_data,
            "graph_queries": graph_queries,
            "text_retrieval": text_retrieval,
        }


def summarize_result(result: dict[str, Any]) -> dict[str, Any]:
    text = result.get("text_retrieval") or {}
    graph_results = [
        relation
        for query in result.get("graph_queries") or []
        for relation in query.get("results") or []
    ]
    return {
        "question": result["question"],
        "route_requested": result["route_requested"],
        "route_used": result["route_used"],
        "fallback_reason": result.get("fallback_reason"),
        "intent": result["intent"],
        "graph_result_count": len(graph_results),
        "graph_targets": [item["target"] for item in graph_results[:20]],
        "text_entity_count": len(text.get("entities") or []),
        "text_relationship_count": len(text.get("relationships") or []),
        "text_chunk_count": len(text.get("chunks") or []),
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="+", help="Medical question to retrieve")
    parser.add_argument("--working-dir", default=WORKING_DIR)
    parser.add_argument("--full", action="store_true", help="Print all evidence")
    args = parser.parse_args()

    pipeline = MedicalQueryPipeline(args.working_dir)
    try:
        result = await pipeline.retrieve(" ".join(args.question))
        output = result if args.full else summarize_result(result)
        print(json.dumps(output, ensure_ascii=False, indent=2))
    finally:
        await pipeline.close()


if __name__ == "__main__":
    asyncio.run(main())
