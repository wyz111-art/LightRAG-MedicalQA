"""Run the versioned medical QA benchmark and write JSON/Markdown reports."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from medical_qa_service import MedicalQAService


DEFAULT_DATASET = Path("medical_evaluation/benchmark_v1.json")
DEFAULT_OUTPUT_DIR = Path("medical_evaluation/reports")


def load_benchmark(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        benchmark = json.load(stream)
    cases = benchmark.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("benchmark must contain a non-empty cases list")
    seen_ids = set()
    for index, case in enumerate(cases, start=1):
        case_id = str(case.get("id") or "").strip()
        question = str(case.get("question") or "").strip()
        expected = case.get("expected")
        if not case_id or not question or not isinstance(expected, dict):
            raise ValueError(f"invalid benchmark case at index {index}")
        if case_id in seen_ids:
            raise ValueError(f"duplicate benchmark case id: {case_id}")
        seen_ids.add(case_id)
    return benchmark


def _recognized_entities(result: dict[str, Any]) -> set[str]:
    return {
        str(item.get("name") or "").strip()
        for item in (result.get("intent") or {}).get("entities") or []
        if str(item.get("name") or "").strip()
    }


def _graph_targets(result: dict[str, Any]) -> set[str]:
    evidence = result.get("graph_evidence")
    if evidence is not None:
        return {
            str(item.get("target") or "").strip()
            for item in evidence
            if str(item.get("target") or "").strip()
        }
    targets = set()
    for query in result.get("graph_queries") or []:
        for item in query.get("results") or []:
            target = str(item.get("target") or "").strip()
            if target:
                targets.add(target)
    return targets


def _evidence_counts(result: dict[str, Any]) -> tuple[int, int]:
    if "graph_evidence" in result:
        return (
            len(result.get("graph_evidence") or []),
            len(result.get("text_evidence") or []),
        )
    graph_count = sum(
        int(query.get("result_count") or 0)
        for query in result.get("graph_queries") or []
    )
    text_count = len((result.get("text_retrieval") or {}).get("chunks") or [])
    return graph_count, text_count


def evaluate_case(
    case: dict[str, Any], result: dict[str, Any], stage: str
) -> dict[str, Any]:
    expected = case["expected"]
    checks = []

    def add_check(name: str, passed: bool, expected_value: Any, actual: Any) -> None:
        checks.append(
            {
                "name": name,
                "passed": bool(passed),
                "expected": expected_value,
                "actual": actual,
            }
        )

    intent = result.get("intent") or {}
    actual_intent = intent.get("intent")
    if expected.get("intent"):
        add_check("intent", actual_intent == expected["intent"], expected["intent"], actual_intent)

    actual_relation = intent.get("relation")
    if "relation" in expected:
        add_check(
            "relation",
            actual_relation == expected["relation"],
            expected["relation"],
            actual_relation,
        )

    routes = expected.get("routes") or []
    if routes:
        actual_route = result.get("route_used")
        add_check("route", actual_route in routes, routes, actual_route)

    expected_entities = set(expected.get("entities") or [])
    if expected_entities:
        actual_entities = _recognized_entities(result)
        add_check(
            "entities",
            expected_entities.issubset(actual_entities),
            sorted(expected_entities),
            sorted(actual_entities),
        )

    expected_targets = set(expected.get("graph_targets_any") or [])
    if expected_targets:
        actual_targets = _graph_targets(result)
        add_check(
            "graph_targets_any",
            bool(expected_targets & actual_targets),
            sorted(expected_targets),
            sorted(actual_targets),
        )

    graph_count, text_count = _evidence_counts(result)
    if "min_graph_evidence" in expected:
        minimum = int(expected["min_graph_evidence"])
        add_check("graph_evidence", graph_count >= minimum, minimum, graph_count)
    if "min_text_evidence" in expected:
        minimum = int(expected["min_text_evidence"])
        add_check("text_evidence", text_count >= minimum, minimum, text_count)

    if stage == "answer":
        answer = str(result.get("answer") or "").strip()
        add_check("answer_nonempty", bool(answer), True, bool(answer))
        required_terms = expected.get("answer_terms_any") or []
        if required_terms:
            add_check(
                "answer_terms_any",
                any(term in answer for term in required_terms),
                required_terms,
                answer,
            )
        forbidden_terms = expected.get("forbidden_answer_terms") or []
        if forbidden_terms:
            found = [term for term in forbidden_terms if term in answer]
            add_check("forbidden_terms", not found, [], found)
        if expected.get("require_citations", True):
            used_ids = set(result.get("used_evidence_ids") or [])
            valid_ids = {
                str(item.get("id"))
                for item in (result.get("graph_evidence") or [])
                + (result.get("text_evidence") or [])
                if item.get("id")
            }
            add_check(
                "citations",
                bool(used_ids) and used_ids.issubset(valid_ids),
                "one or more valid evidence IDs",
                sorted(used_ids),
            )

    passed_count = sum(check["passed"] for check in checks)
    score = passed_count / len(checks) if checks else 1.0
    return {
        "id": case["id"],
        "department": case.get("department"),
        "category": case.get("category"),
        "question": case["question"],
        "passed": all(check["passed"] for check in checks),
        "score": round(score, 4),
        "checks": checks,
        "timings": result.get("timings") or {},
        "result_summary": {
            "intent": actual_intent,
            "relation": actual_relation,
            "route": result.get("route_used"),
            "entities": sorted(_recognized_entities(result)),
            "graph_targets": sorted(_graph_targets(result)),
            "graph_evidence_count": graph_count,
            "text_evidence_count": text_count,
            "answer": result.get("answer") if stage == "answer" else None,
            "used_evidence_ids": result.get("used_evidence_ids") or [],
        },
    }


def summarize(
    benchmark: dict[str, Any], results: list[dict[str, Any]], stage: str
) -> dict[str, Any]:
    metric_totals = Counter()
    metric_passed = Counter()
    latencies = []
    for result in results:
        for check in result.get("checks") or []:
            metric_totals[check["name"]] += 1
            metric_passed[check["name"]] += int(check["passed"])
        total_ms = (result.get("timings") or {}).get("total_ms")
        if isinstance(total_ms, (int, float)):
            latencies.append(float(total_ms))

    scores = [float(result["score"]) for result in results]
    sorted_latencies = sorted(latencies)
    p95_index = max(0, math.ceil(len(sorted_latencies) * 0.95) - 1)
    return {
        "benchmark_version": benchmark.get("version"),
        "stage": stage,
        "case_count": len(results),
        "passed_cases": sum(result["passed"] for result in results),
        "pass_rate": round(
            sum(result["passed"] for result in results) / max(len(results), 1), 4
        ),
        "average_score": round(statistics.mean(scores), 4) if scores else 0.0,
        "metrics": {
            name: {
                "passed": metric_passed[name],
                "total": total,
                "accuracy": round(metric_passed[name] / total, 4),
            }
            for name, total in sorted(metric_totals.items())
        },
        "latency_ms": {
            "mean": round(statistics.mean(latencies), 1) if latencies else None,
            "median": round(statistics.median(latencies), 1) if latencies else None,
            "p95": round(sorted_latencies[p95_index], 1) if latencies else None,
        },
    }


def markdown_report(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        f"# 医疗问答评测报告（{summary['benchmark_version']}）",
        "",
        f"- 评测阶段：`{summary['stage']}`",
        f"- 用例数量：{summary['case_count']}",
        f"- 完全通过：{summary['passed_cases']}",
        f"- 完全通过率：{summary['pass_rate']:.1%}",
        f"- 平均得分：{summary['average_score']:.1%}",
        "",
        "## 指标",
        "",
        "| 指标 | 通过 | 总数 | 准确率 |",
        "|---|---:|---:|---:|",
    ]
    for name, metric in summary["metrics"].items():
        lines.append(
            f"| {name} | {metric['passed']} | {metric['total']} | "
            f"{metric['accuracy']:.1%} |"
        )
    latency = summary["latency_ms"]
    lines.extend(
        [
            "",
            "## 延迟",
            "",
            f"- 平均：{latency['mean']} ms",
            f"- 中位数：{latency['median']} ms",
            f"- P95：{latency['p95']} ms",
            "",
            "## 用例明细",
            "",
            "| 用例 | 分类 | 得分 | 结果 | 失败检查 |",
            "|---|---|---:|---|---|",
        ]
    )
    for result in report["results"]:
        failed = ", ".join(
            check["name"] for check in result["checks"] if not check["passed"]
        )
        lines.append(
            f"| {result['id']} | {result.get('category') or ''} | "
            f"{result['score']:.1%} | {'通过' if result['passed'] else '未通过'} | "
            f"{failed or '-'} |"
        )
    return "\n".join(lines) + "\n"


def select_cases(
    benchmark: dict[str, Any], case_ids: list[str], limit: int
) -> list[dict[str, Any]]:
    cases = benchmark["cases"]
    if case_ids:
        requested = set(case_ids)
        cases = [case for case in cases if case["id"] in requested]
        missing = requested - {case["id"] for case in cases}
        if missing:
            raise ValueError(f"unknown case IDs: {sorted(missing)}")
    if limit > 0:
        cases = cases[:limit]
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--working-dir", default="medical_kg")
    parser.add_argument("--stage", choices=["retrieval", "answer"], default="retrieval")
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--fail-below", type=float, default=0.0)
    args = parser.parse_args()

    benchmark = load_benchmark(args.dataset)
    cases = select_cases(benchmark, args.case_id, args.limit)
    print(
        f"Benchmark {benchmark.get('version')} loaded: {len(cases)} / "
        f"{len(benchmark['cases'])} cases"
    )
    if args.validate_only:
        counts = Counter(case.get("category") for case in cases)
        print(json.dumps({"categories": counts}, ensure_ascii=False, indent=2))
        return 0

    service = MedicalQAService(args.working_dir)
    evaluated = []
    try:
        for index, case in enumerate(cases, start=1):
            print(f"[{index}/{len(cases)}] {case['id']}: {case['question']}")
            try:
                if args.stage == "answer":
                    result = service.answer(case["question"])
                else:
                    result = service.retrieve(case["question"])
                evaluated.append(evaluate_case(case, result, args.stage))
            except Exception as exc:
                evaluated.append(
                    {
                        "id": case["id"],
                        "department": case.get("department"),
                        "category": case.get("category"),
                        "question": case["question"],
                        "passed": False,
                        "score": 0.0,
                        "checks": [
                            {
                                "name": "runtime_error",
                                "passed": False,
                                "expected": "successful execution",
                                "actual": f"{type(exc).__name__}: {exc}",
                            }
                        ],
                        "timings": {},
                        "result_summary": {},
                    }
                )
    finally:
        service.close()

    report = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "dataset": str(args.dataset),
        "summary": summarize(benchmark, evaluated, args.stage),
        "results": evaluated,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base_name = f"medical-eval-{args.stage}-{timestamp}"
    json_path = args.output_dir / f"{base_name}.json"
    markdown_path = args.output_dir / f"{base_name}.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    markdown_path.write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"JSON report: {json_path.resolve()}")
    print(f"Markdown report: {markdown_path.resolve()}")
    return int(report["summary"]["average_score"] < args.fail_below)


if __name__ == "__main__":
    sys.exit(main())
