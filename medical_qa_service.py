"""Long-lived medical QA service backed by one dedicated asyncio event loop."""

from __future__ import annotations

import asyncio
import atexit
import concurrent.futures
import os
import threading
import time
from collections.abc import Callable
from typing import Any

from medical_answer import MedicalAnswerGenerator
from medical_query_pipeline import MedicalQueryPipeline


PipelineFactory = Callable[[], MedicalQueryPipeline]
GeneratorFactory = Callable[[], MedicalAnswerGenerator]


class MedicalQAService:
    """Keep LightRAG, Ollama clients, and storages alive across many queries.

    All async resources are created, used, and closed on the same background
    event loop. Calls are serialized because the file-backed LightRAG instance
    and its worker queues are shared by every Streamlit session in the process.
    """

    def __init__(
        self,
        working_dir: str = "medical_kg",
        *,
        request_timeout: float | None = None,
        pipeline_factory: PipelineFactory | None = None,
        generator_factory: GeneratorFactory | None = None,
    ) -> None:
        self.working_dir = working_dir
        self.request_timeout = request_timeout or float(
            os.getenv("MEDICAL_QUERY_TIMEOUT", "900")
        )
        self._pipeline_factory = pipeline_factory or (
            lambda: MedicalQueryPipeline(self.working_dir)
        )
        self._generator_factory = generator_factory or MedicalAnswerGenerator
        self._thread = threading.Thread(
            target=self._run_event_loop,
            name="medical-qa-service",
            daemon=True,
        )
        self._loop_ready = threading.Event()
        self._close_lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._pipeline: MedicalQueryPipeline | None = None
        self._generator: MedicalAnswerGenerator | None = None
        self._query_lock: asyncio.Lock | None = None
        self._closed = False

        self._thread.start()
        if not self._loop_ready.wait(timeout=10):
            raise RuntimeError("Medical QA event loop failed to start")
        try:
            self._submit(self._initialize())
        except Exception:
            self.close()
            raise
        atexit.register(self.close)

    def _run_event_loop(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        self._loop_ready.set()
        try:
            loop.run_forever()
        finally:
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()

    async def _initialize(self) -> None:
        self._pipeline = self._pipeline_factory()
        self._generator = self._generator_factory()
        self._query_lock = asyncio.Lock()
        initialize = getattr(self._pipeline, "initialize", None)
        if initialize is not None:
            await initialize()

    def _submit(self, coroutine):
        if self._closed:
            coroutine.close()
            raise RuntimeError("Medical QA service is closed")
        loop = self._loop
        if loop is None or loop.is_closed():
            coroutine.close()
            raise RuntimeError("Medical QA event loop is unavailable")
        future = asyncio.run_coroutine_threadsafe(coroutine, loop)
        try:
            return future.result(timeout=self.request_timeout)
        except concurrent.futures.TimeoutError as exc:
            future.cancel()
            raise TimeoutError(
                f"Medical query exceeded {self.request_timeout:.0f} seconds"
            ) from exc

    async def _retrieve(self, question: str, submitted_at: float) -> dict[str, Any]:
        if self._pipeline is None or self._query_lock is None:
            raise RuntimeError("Medical QA service is not initialized")
        async with self._query_lock:
            started_at = time.perf_counter()
            result = await self._pipeline.retrieve(question)
            finished_at = time.perf_counter()
        result = dict(result)
        result["timings"] = {
            "queue_wait_ms": round((started_at - submitted_at) * 1000, 1),
            "retrieval_ms": round((finished_at - started_at) * 1000, 1),
            "generation_ms": 0.0,
            "total_ms": round((finished_at - submitted_at) * 1000, 1),
        }
        return result

    async def _answer(self, question: str, submitted_at: float) -> dict[str, Any]:
        if (
            self._pipeline is None
            or self._generator is None
            or self._query_lock is None
        ):
            raise RuntimeError("Medical QA service is not initialized")
        async with self._query_lock:
            started_at = time.perf_counter()
            retrieval = await self._pipeline.retrieve(question)
            retrieved_at = time.perf_counter()
            result = await self._generator.generate(question, retrieval)
            finished_at = time.perf_counter()
        result = dict(result)
        result["timings"] = {
            "queue_wait_ms": round((started_at - submitted_at) * 1000, 1),
            "retrieval_ms": round((retrieved_at - started_at) * 1000, 1),
            "generation_ms": round((finished_at - retrieved_at) * 1000, 1),
            "total_ms": round((finished_at - submitted_at) * 1000, 1),
        }
        return result

    def retrieve(self, question: str) -> dict[str, Any]:
        question = question.strip()
        if not question:
            raise ValueError("question cannot be empty")
        return self._submit(self._retrieve(question, time.perf_counter()))

    def answer(self, question: str) -> dict[str, Any]:
        question = question.strip()
        if not question:
            raise ValueError("question cannot be empty")
        return self._submit(self._answer(question, time.perf_counter()))

    async def _close_async(self) -> None:
        try:
            if self._pipeline is not None:
                await self._pipeline.close()
                self._pipeline = None
        finally:
            if self._generator is not None:
                await self._generator.close()
                self._generator = None

    def close(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            loop = self._loop
            if loop is not None and not loop.is_closed():
                try:
                    future = asyncio.run_coroutine_threadsafe(
                        self._close_async(), loop
                    )
                    future.result(timeout=min(self.request_timeout, 30))
                except Exception:
                    pass
                loop.call_soon_threadsafe(loop.stop)
            self._thread.join(timeout=10)
            self._closed = True

    @property
    def is_running(self) -> bool:
        return not self._closed and self._thread.is_alive()
