"""LLM-based intent and medical entity recognition for query routing."""

import argparse
import asyncio
import json
import os
import re
from dataclasses import asdict, dataclass
from typing import Any

import ollama
from json_repair import repair_json


INTENT_CONFIG = {
    "query_symptoms": {
        "label": "查询症状",
        "relation": "has_symptom",
        "mode": "graph",
    },
    "query_drugs": {
        "label": "查询药物",
        "relation": "treats",
        "mode": "graph",
    },
    "query_causes": {
        "label": "查询病因",
        "relation": "causes",
        "mode": "graph",
    },
    "query_complications": {
        "label": "查询并发症",
        "relation": "complication",
        "mode": "graph",
    },
    "query_exams": {
        "label": "查询检查",
        "relation": "exam",
        "mode": "graph",
    },
    "query_contraindications": {
        "label": "查询禁忌",
        "relation": "contraindicates",
        "mode": "graph",
    },
    "query_department": {
        "label": "查询科室",
        "relation": None,
        "mode": "hybrid",
    },
    "disease_overview": {
        "label": "疾病介绍",
        "relation": None,
        "mode": "hybrid",
    },
    "treatment_advice": {
        "label": "治疗方法",
        "relation": "treats",
        "mode": "text",
    },
    "precautions": {
        "label": "注意事项",
        "relation": None,
        "mode": "text",
    },
    "other": {
        "label": "其他或无法判断",
        "relation": None,
        "mode": "hybrid",
    },
}

ENTITY_TYPES = {
    "Disease",
    "Drug",
    "Symptom",
    "BodyPart",
    "Exam",
    "Food",
    "Department",
    "Other",
}

SYSTEM_PROMPT = """你是医疗问答系统的意图识别器，只负责分类和实体提取，不回答医疗问题。

必须从以下 intent 中选择一个：
- query_symptoms：查询疾病症状
- query_drugs：查询疾病相关药物
- query_causes：查询疾病病因或诱因
- query_complications：查询疾病并发症
- query_exams：查询疾病相关检查
- query_contraindications：查询药物禁忌或某疾病是否禁用某药
- query_department：查询应该就诊的科室
- disease_overview：询问疾病是什么或基本介绍
- treatment_advice：询问治疗方法、手术或用药方案
- precautions：询问饮食、生活方式、护理或注意事项
- other：不属于以上类别或无法确定

实体类型只能使用：Disease、Drug、Symptom、BodyPart、Exam、Food、Department、Other。
只输出一个 JSON 对象，不要输出 Markdown、解释、诊断或建议。

输出格式：
{
  "intent": "query_complications",
  "entities": [{"name": "高血压", "type": "Disease"}],
  "confidence": 0.92
}

示例：
问题：高血压会引起哪些并发症？
输出：{"intent":"query_complications","entities":[{"name":"高血压","type":"Disease"}],"confidence":0.95}

问题：肺炎患者平时需要注意什么？
输出：{"intent":"precautions","entities":[{"name":"肺炎","type":"Disease"}],"confidence":0.93}
"""


@dataclass(frozen=True)
class IntentResult:
    intent: str
    intent_label: str
    entities: list[dict[str, str]]
    relation: str | None
    retrieval_mode: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _extract_json_text(value: str) -> str:
    text = value.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start : end + 1]
    return text


def _normalize_entities(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []

    entities = []
    seen = set()
    for item in value:
        if isinstance(item, str):
            name = item.strip()
            entity_type = "Other"
        elif isinstance(item, dict):
            name = str(item.get("name") or "").strip()
            entity_type = str(item.get("type") or "Other").strip()
        else:
            continue

        if not name or name in seen:
            continue
        if entity_type not in ENTITY_TYPES:
            entity_type = "Other"
        seen.add(name)
        entities.append({"name": name, "type": entity_type})
    return entities


def normalize_intent_payload(payload: Any) -> IntentResult:
    if not isinstance(payload, dict):
        payload = {}

    intent = str(payload.get("intent") or "other").strip()
    if intent not in INTENT_CONFIG:
        intent = "other"
    config = INTENT_CONFIG[intent]

    try:
        confidence = float(payload.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    retrieval_mode = config["mode"]
    if retrieval_mode == "graph" and confidence < 0.65:
        retrieval_mode = "hybrid"

    return IntentResult(
        intent=intent,
        intent_label=config["label"],
        entities=_normalize_entities(payload.get("entities")),
        relation=config["relation"],
        retrieval_mode=retrieval_mode,
        confidence=confidence,
    )


def parse_intent_response(content: str) -> IntentResult:
    json_text = _extract_json_text(content)
    try:
        payload = json.loads(json_text)
    except json.JSONDecodeError:
        payload = json.loads(repair_json(json_text))
    return normalize_intent_payload(payload)


class MedicalIntentRecognizer:
    def __init__(self, model: str | None = None, host: str | None = None):
        self.model = model or os.getenv("MEDICAL_INTENT_MODEL", "qwen2.5:14b")
        ollama_host = host or os.getenv("OLLAMA_HOST")
        self.client = (
            ollama.AsyncClient(host=ollama_host)
            if ollama_host
            else ollama.AsyncClient()
        )

    async def recognize(self, question: str) -> IntentResult:
        question = question.strip()
        if not question:
            raise ValueError("question cannot be empty")

        response = await self.client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            format="json",
            options={"temperature": 0},
        )
        return parse_intent_response(response["message"]["content"])

    async def close(self) -> None:
        await self.client.close()


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="+", help="Medical question to classify")
    parser.add_argument("--model", help="Ollama model, defaults to MEDICAL_INTENT_MODEL")
    args = parser.parse_args()

    recognizer = MedicalIntentRecognizer(model=args.model)
    try:
        result = await recognizer.recognize(" ".join(args.question))
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    finally:
        await recognizer.close()


if __name__ == "__main__":
    asyncio.run(main())
