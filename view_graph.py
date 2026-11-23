"""Generate the interactive Chinese UI for the complete medical graph."""

import os
from pathlib import Path

import networkx as nx

from medical_graph_ui import build_full_graph_html


WORKING_DIR = Path(os.getenv("MEDICAL_WORKING_DIR", "medical_kg"))
GRAPH = WORKING_DIR / "graph_chunk_entity_relation.graphml"
OUT = WORKING_DIR / "graph_view.html"
MAX_NODES = int(os.getenv("MEDICAL_GRAPH_MAX_NODES", "0"))
MIN_DEGREE = int(os.getenv("MEDICAL_GRAPH_MIN_DEGREE", "0"))
LABEL_COUNT = int(os.getenv("MEDICAL_GRAPH_LABEL_COUNT", "80"))


def main() -> None:
    if not GRAPH.is_file():
        raise FileNotFoundError(f"Graph not found: {GRAPH}. Run build_kg.py first.")
    if MAX_NODES < 0 or MIN_DEGREE < 0 or LABEL_COUNT < 1:
        raise ValueError("graph display limits must be non-negative")

    graph = nx.read_graphml(GRAPH)
    print(
        f"Loading complete graph: {graph.number_of_nodes()} nodes, "
        f"{graph.number_of_edges()} edges"
    )
    graph_html = build_full_graph_html(
        graph,
        max_nodes=MAX_NODES,
        min_degree=MIN_DEGREE,
        label_count=LABEL_COUNT,
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(graph_html, encoding="utf-8")
    print(f"Saved interactive graph: {OUT.resolve()}")
    print(f"HTML size: {OUT.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
