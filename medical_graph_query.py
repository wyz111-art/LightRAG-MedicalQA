"""Read-only relation retrieval over the existing medical LightRAG graph."""

import argparse
import json
import re
from pathlib import Path
from typing import Any

import networkx as nx

from lightrag.constants import GRAPH_FIELD_SEP


ALLOWED_RELATIONS = {
    "treats",
    "has_symptom",
    "causes",
    "complication",
    "exam",
    "contraindicates",
}

RELATION_TARGET_TYPES = {
    "has_symptom": {"symptom"},
    "complication": {"disease"},
    "exam": {"exam", "examination"},
}


def split_graph_field(value: Any) -> list[str]:
    items = []
    for item in str(value or "").split(GRAPH_FIELD_SEP):
        cleaned = item.strip()
        if cleaned and cleaned not in items:
            items.append(cleaned)
    return items


def split_relation_keywords(value: Any) -> list[str]:
    keywords = []
    for item in re.split(r"<SEP>|[,，;；|/]+", str(value or "")):
        cleaned = item.strip().lower()
        if cleaned and cleaned not in keywords:
            keywords.append(cleaned)
    return keywords


class MedicalGraphRetriever:
    def __init__(self, working_dir: str | Path = "medical_kg"):
        self.working_dir = Path(working_dir)
        graph_path = self.working_dir / "graph_chunk_entity_relation.graphml"
        chunks_path = self.working_dir / "kv_store_text_chunks.json"
        if not graph_path.is_file():
            raise FileNotFoundError(f"Graph not found: {graph_path}")
        if not chunks_path.is_file():
            raise FileNotFoundError(f"Text chunk storage not found: {chunks_path}")

        self.graph = nx.read_graphml(graph_path)
        with chunks_path.open(encoding="utf-8") as stream:
            self.text_chunks = json.load(stream)
        self._casefold_nodes = {
            str(node).strip().casefold(): str(node) for node in self.graph.nodes
        }

    def match_entity(self, entity_name: str) -> dict[str, Any]:
        query = entity_name.strip()
        if not query:
            return {"matched": None, "match_type": "empty", "suggestions": []}
        if self.graph.has_node(query):
            return {"matched": query, "match_type": "exact", "suggestions": []}

        casefold_match = self._casefold_nodes.get(query.casefold())
        if casefold_match:
            return {
                "matched": casefold_match,
                "match_type": "case_insensitive",
                "suggestions": [],
            }

        contained_aliases = [
            str(node)
            for node in self.graph.nodes
            if len(str(node).strip()) >= 2 and str(node).strip() in query
        ]
        if contained_aliases:
            longest_length = max(len(node) for node in contained_aliases)
            longest_aliases = sorted(
                {node for node in contained_aliases if len(node) == longest_length},
                key=lambda node: (-self.graph.degree(node), node),
            )
            if len(longest_aliases) == 1:
                return {
                    "matched": longest_aliases[0],
                    "match_type": "contained_alias",
                    "suggestions": [],
                }

        suggestions = [
            str(node)
            for node in self.graph.nodes
            if query in str(node) or str(node) in query
        ]
        suggestions.sort(
            key=lambda node: (
                abs(len(node) - len(query)),
                -self.graph.degree(node),
                node,
            )
        )
        return {
            "matched": None,
            "match_type": "not_found",
            "suggestions": suggestions[:10],
        }

    def _chunk_evidence(
        self,
        source_ids: list[str],
        limit: int,
        required_terms: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        candidates = []
        seen_content = set()
        for chunk_id in source_ids:
            chunk = self.text_chunks.get(chunk_id)
            if not chunk:
                continue
            content = str(chunk.get("content") or "").strip()
            if not content or content in seen_content:
                continue
            seen_content.add(content)
            matched_terms = [
                term for term in (required_terms or []) if term and term in content
            ]
            candidates.append(
                {
                    "chunk_id": chunk_id,
                    "content": content,
                    "file_path": chunk.get("file_path", "unknown_source"),
                    "matched_terms": matched_terms,
                    "supports_relation": bool(required_terms)
                    and len(matched_terms) == len(required_terms),
                }
            )
        candidates.sort(
            key=lambda item: (
                not item["supports_relation"],
                -len(item["matched_terms"]),
                item["chunk_id"],
            )
        )
        return candidates[:limit]

    def query_relation(
        self,
        entity_name: str,
        relation: str,
        *,
        limit: int = 20,
        evidence_per_relation: int = 2,
        _allow_relation_alias_fallback: bool = True,
    ) -> dict[str, Any]:
        relation = relation.strip().lower()
        if relation not in ALLOWED_RELATIONS:
            raise ValueError(
                f"Unsupported relation {relation!r}; expected one of "
                f"{sorted(ALLOWED_RELATIONS)}"
            )
        if limit < 1 or evidence_per_relation < 0:
            raise ValueError("limits must be positive")

        entity_match = self.match_entity(entity_name)
        matched_entity = entity_match["matched"]
        if not matched_entity:
            return {
                "status": "entity_not_found",
                "query_entity": entity_name,
                "relation": relation,
                "entity_match": entity_match,
                "results": [],
                "subgraph": {"nodes": [], "edges": []},
            }

        source_type = str(
            self.graph.nodes[matched_entity].get("entity_type", "UNKNOWN")
        ).lower()
        allowed_target_types = RELATION_TARGET_TYPES.get(relation)
        if relation in {"treats", "contraindicates"}:
            if source_type in {"drug", "medication"}:
                allowed_target_types = {"disease"}
            elif source_type == "disease":
                allowed_target_types = {"drug", "medication"}

        results = []
        filtered_by_type = 0
        for neighbor in self.graph.neighbors(matched_entity):
            edge_data = dict(self.graph.get_edge_data(matched_entity, neighbor) or {})
            keywords = split_relation_keywords(edge_data.get("keywords"))
            if relation not in keywords:
                continue

            source_ids = split_graph_field(edge_data.get("source_id"))
            neighbor_data = dict(self.graph.nodes[neighbor])
            target_type = str(neighbor_data.get("entity_type", "UNKNOWN"))
            if (
                allowed_target_types
                and target_type.lower() not in allowed_target_types
            ):
                filtered_by_type += 1
                continue
            try:
                weight = float(edge_data.get("weight", 1.0))
            except (TypeError, ValueError):
                weight = 1.0
            results.append(
                {
                    "source": matched_entity,
                    "target": str(neighbor),
                    "target_type": target_type,
                    "relation": relation,
                    "all_keywords": keywords,
                    "weight": weight,
                    "descriptions": split_graph_field(edge_data.get("description")),
                    "source_ids": source_ids,
                    "evidence": self._chunk_evidence(
                        source_ids,
                        evidence_per_relation,
                        required_terms=[matched_entity, str(neighbor)],
                    ),
                }
            )

        results.sort(key=lambda item: (-item["weight"], item["target"]))
        results = results[:limit]
        if not results and _allow_relation_alias_fallback:
            relation_aliases = [
                str(node)
                for node in self.graph.nodes
                if str(node) != matched_entity
                and len(str(node).strip()) >= 2
                and str(node).strip() in entity_name
            ]
            if relation_aliases:
                longest_length = max(len(node) for node in relation_aliases)
                longest_aliases = sorted(
                    {
                        node
                        for node in relation_aliases
                        if len(node) == longest_length
                    },
                    key=lambda node: (-self.graph.degree(node), node),
                )
                if len(longest_aliases) == 1:
                    alias = longest_aliases[0]
                    fallback = self.query_relation(
                        alias,
                        relation,
                        limit=limit,
                        evidence_per_relation=evidence_per_relation,
                        _allow_relation_alias_fallback=False,
                    )
                    if fallback.get("result_count", 0):
                        fallback["query_entity"] = entity_name
                        fallback["entity_match"] = {
                            "matched": alias,
                            "match_type": "relation_alias",
                            "suggestions": [],
                        }
                        return fallback
        nodes = [
            {
                "id": matched_entity,
                "label": matched_entity,
                "entity_type": str(
                    self.graph.nodes[matched_entity].get("entity_type", "UNKNOWN")
                ),
            }
        ]
        nodes.extend(
            {
                "id": item["target"],
                "label": item["target"],
                "entity_type": item["target_type"],
            }
            for item in results
        )
        edges = [
            {
                "source": item["source"],
                "target": item["target"],
                "relation": item["relation"],
                "weight": item["weight"],
            }
            for item in results
        ]
        return {
            "status": "success" if results else "relation_not_found",
            "query_entity": entity_name,
            "matched_entity": matched_entity,
            "relation": relation,
            "graph_is_directed": self.graph.is_directed(),
            "entity_match": entity_match,
            "result_count": len(results),
            "filtered_by_type": filtered_by_type,
            "results": results,
            "subgraph": {"nodes": nodes, "edges": edges},
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("entity", help="Entity name, for example 高血压")
    parser.add_argument("relation", choices=sorted(ALLOWED_RELATIONS))
    parser.add_argument("--working-dir", default="medical_kg")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--evidence-per-relation", type=int, default=2)
    args = parser.parse_args()

    retriever = MedicalGraphRetriever(args.working_dir)
    result = retriever.query_relation(
        args.entity,
        args.relation,
        limit=args.limit,
        evidence_per_relation=args.evidence_per_relation,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
