"""Reusable Chinese graph UI for the medical KG and QA result subgraphs."""

from __future__ import annotations

import hashlib
import html
import json
import math
import re
from collections import Counter
from typing import Any

import networkx as nx
from pyvis.network import Network


TYPE_COLORS = {
    "disease": "#EF4444",
    "drug": "#1683E7",
    "symptom": "#F07B26",
    "bodypart": "#8757E8",
    "exam": "#00A66A",
    "food": "#E4AA00",
    "department": "#77818C",
    "other": "#9AA3AD",
}
TYPE_LABELS = {
    "disease": "疾病",
    "drug": "药物",
    "symptom": "症状",
    "bodypart": "身体部位",
    "exam": "检查",
    "food": "食物",
    "department": "科室",
    "other": "其他",
}
RELATION_LABELS = {
    "treats": "治疗药物",
    "has_symptom": "相关症状",
    "causes": "病因/导致",
    "complication": "并发症",
    "exam": "检查项目",
    "contraindicates": "用药禁忌",
    "治疗药物": "治疗药物",
    "相关症状": "相关症状",
    "病因": "病因/导致",
    "并发症": "并发症",
    "检查": "检查项目",
    "禁忌": "用药禁忌",
    "相关科室": "相关科室",
}

_TYPE_ALIASES = {"medication": "drug", "examination": "exam"}
_TYPE_ORDER = [
    "disease",
    "symptom",
    "drug",
    "exam",
    "bodypart",
    "food",
    "department",
    "other",
]
_FIELD_SEP_RE = re.compile(r"<\s*SEP\s*>", re.IGNORECASE)


def normalize_entity_type(value: Any) -> str:
    entity_type = str(value or "other").strip().lower()
    entity_type = _TYPE_ALIASES.get(entity_type, entity_type)
    return entity_type if entity_type in TYPE_LABELS else "other"


def relation_label(value: Any) -> str:
    labels = []
    for part in re.split(r"<SEP>|[,，;；|/]+", str(value or "")):
        keyword = part.strip()
        canonical = keyword.split("(", 1)[0].split("（", 1)[0].strip().lower()
        label = RELATION_LABELS.get(canonical) or RELATION_LABELS.get(keyword)
        if label and label not in labels:
            labels.append(label)
    return "、".join(labels) or "其他关系"


def compact_text(value: Any, limit: int = 240) -> str:
    text = html.unescape(str(value or ""))
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = re.sub(r"(?i)</?(?:b|strong|i|em)>", "", text)
    text = _FIELD_SEP_RE.sub("\n", text).replace("\\n", "\n")
    items: list[str] = []
    for part in text.splitlines():
        cleaned = re.sub(r"\s+", " ", part).strip(" \t-;；")
        if cleaned and cleaned not in items:
            items.append(cleaned)
    combined = "；".join(items)
    return combined if len(combined) <= limit else f"{combined[: limit - 1]}…"


def _source_count(value: Any) -> int:
    return len([item for item in _FIELD_SEP_RE.split(str(value or "")) if item.strip()])


def _initial_position(value: str) -> tuple[float, float]:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=16).digest()
    angle_value = int.from_bytes(digest[:8], "big") / (2**64)
    radius_value = int.from_bytes(digest[8:], "big") / (2**64)
    angle = angle_value * math.tau
    radius = 3400 * math.sqrt(radius_value)
    return 1.25 * radius * math.cos(angle), radius * math.sin(angle)


def _full_layout(
    node_ids: list[str], graph: nx.Graph
) -> dict[str, tuple[int, int]]:
    """Approximate a force layout in linear time without forming type clusters."""
    index_by_node = {node_id: index for index, node_id in enumerate(node_ids)}
    initial = [_initial_position(node_id) for node_id in node_ids]
    x_values = [position[0] for position in initial]
    y_values = [position[1] for position in initial]
    edge_indexes = [
        (index_by_node[str(source)], index_by_node[str(target)])
        for source, target in graph.edges()
        if str(source) in index_by_node and str(target) in index_by_node
    ]

    for _iteration in range(9):
        x_sums = [0.0] * len(node_ids)
        y_sums = [0.0] * len(node_ids)
        counts = [0] * len(node_ids)
        for source_index, target_index in edge_indexes:
            x_sums[source_index] += x_values[target_index]
            y_sums[source_index] += y_values[target_index]
            counts[source_index] += 1
            x_sums[target_index] += x_values[source_index]
            y_sums[target_index] += y_values[source_index]
            counts[target_index] += 1

        next_x = x_values.copy()
        next_y = y_values.copy()
        for index, count in enumerate(counts):
            if not count:
                continue
            neighbor_x = x_sums[index] / count
            neighbor_y = y_sums[index] / count
            next_x[index] = (
                0.52 * x_values[index]
                + 0.26 * neighbor_x
                + 0.22 * initial[index][0]
            )
            next_y[index] = (
                0.52 * y_values[index]
                + 0.26 * neighbor_y
                + 0.22 * initial[index][1]
            )
        x_values, y_values = next_x, next_y

    return {
        node_id: (round(x_values[index]), round(y_values[index]))
        for index, node_id in enumerate(node_ids)
    }


def _query_layout(
    nodes: list[dict[str, Any]], edges: list[dict[str, Any]]
) -> dict[str, tuple[int, int]]:
    node_ids = [str(node.get("id")) for node in nodes if node.get("id")]
    if not node_ids:
        return {}
    degree = Counter()
    for edge in edges:
        degree[str(edge.get("source"))] += 1
        degree[str(edge.get("target"))] += 1
    center = max(node_ids, key=lambda node_id: (degree[node_id], -node_ids.index(node_id)))
    positions = {center: (0, 0)}
    others = [node_id for node_id in node_ids if node_id != center]
    for index, node_id in enumerate(others):
        angle = math.tau * index / max(len(others), 1) - math.pi / 2
        radius = 250 + 32 * (index % 3)
        positions[node_id] = (round(radius * math.cos(angle)), round(radius * math.sin(angle)))
    return positions


UI_CSS = r"""
<style>
html,body{width:100%;height:100%;margin:0;overflow:hidden;background:#f7f9fb;color:#20272e;font-family:"Microsoft YaHei","PingFang SC",Arial,sans-serif}
*{box-sizing:border-box}button,input{font:inherit}
#kg-app{height:100vh;min-height:560px;display:flex;flex-direction:column;background:#f7f9fb}
.kg-head{height:56px;display:flex;align-items:center;gap:12px;padding:0 16px;background:#fff;border-bottom:1px solid #e5e9ee;flex:none}
.kg-mark{width:31px;height:31px;display:grid;place-items:center;border-radius:7px;background:#ef4444;color:#fff;font-size:17px;font-weight:800}.kg-title{font-size:16px;font-weight:750}.kg-sub{font-size:11px;color:#7b8794;margin-top:2px;max-width:520px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.kg-grow{flex:1}
.kg-btn{height:34px;border:1px solid #d8dee5;background:#fff;color:#26313b;border-radius:6px;padding:0 11px;display:inline-flex;align-items:center;gap:6px}.kg-btn:hover{border-color:#9ba8b5;background:#f8fafc}.kg-btn.primary{background:#ef4444;border-color:#ef4444;color:#fff}
.kg-toolbar{height:52px;display:flex;align-items:center;gap:10px;padding:8px 14px;border-bottom:1px solid #e5e9ee;flex:none}.kg-search{position:relative;width:min(390px,42%)}.kg-search input{width:100%;height:36px;border:1px solid #ccd5df;background:#fff;border-radius:6px;padding:0 66px 0 12px;outline:none}.kg-search input:focus{border-color:#ef4444;box-shadow:0 0 0 2px rgba(239,68,68,.12)}.kg-search button{position:absolute;right:4px;top:4px;height:28px;border:0;border-radius:5px;background:#ef4444;color:#fff;padding:0 11px}
.kg-segment{height:36px;display:flex;gap:2px;padding:3px;border-radius:6px;background:#edf1f5}.kg-segment button{border:0;border-radius:4px;background:transparent;color:#6f7b87;padding:0 11px}.kg-segment button.active{background:#fff;color:#20272e;box-shadow:0 1px 2px rgba(26,34,42,.12)}.kg-mode-label{height:30px;display:flex;align-items:center;padding:0 10px;border-radius:5px;background:#edf1f5;color:#53606c;font-size:12px}
.kg-body{min-height:0;flex:1;display:grid;grid-template-columns:188px minmax(0,1fr) 238px}.kg-side{min-height:0;overflow:auto;background:#fff;padding:14px 12px}.kg-side.left{border-right:1px solid #e5e9ee}.kg-side.right{border-left:1px solid #e5e9ee}.kg-section-title{display:flex;justify-content:space-between;font-size:12px;font-weight:700;margin-bottom:9px}.kg-section-title span{font-size:10px;font-weight:500;color:#8a96a2}
.kg-filter{width:100%;display:flex;align-items:center;gap:8px;border:0;background:transparent;color:#26313b;padding:7px 5px;border-radius:5px;text-align:left}.kg-filter:hover{background:#f3f6f8}.kg-filter.off{opacity:.34}.kg-dot{width:9px;height:9px;border-radius:50%;background:var(--type-color)}.kg-filter .name{flex:1;font-size:12px}.kg-filter .count{font-size:10px;color:#8a96a2}
.kg-main{position:relative;min-width:0;min-height:0;background:#f7f9fb;overflow:hidden}#mynetwork{position:absolute!important;inset:0;width:100%!important;height:100%!important;border:0!important;background:#f7f9fb!important}.kg-status{position:absolute;left:13px;top:11px;z-index:4;padding:5px 8px;border:1px solid #dce2e8;border-radius:5px;background:rgba(255,255,255,.92);color:#697581;font-size:11px;pointer-events:none}.kg-controls{position:absolute;right:12px;bottom:12px;z-index:4;display:grid;gap:6px}.kg-controls button{width:34px;height:34px;border:1px solid #d8dee5;border-radius:6px;background:#fff;color:#26313b;font-size:17px}.kg-controls button:hover{border-color:#9ba8b5}.kg-legend{position:absolute;left:13px;bottom:12px;z-index:4;padding:6px 9px;border:1px solid #dce2e8;border-radius:5px;background:rgba(255,255,255,.92);color:#697581;font-size:10px}
.kg-type{display:inline-flex;align-items:center;gap:6px;padding:4px 7px;border-radius:5px;background:#edf1f5;color:#687582;font-size:11px;margin-bottom:10px}.kg-type i{width:8px;height:8px;border-radius:50%;background:var(--detail-color)}.kg-name{font-size:19px;font-weight:750;margin-bottom:6px;word-break:break-word}.kg-desc{font-size:12px;line-height:1.65;color:#66737f;margin-bottom:14px;white-space:pre-wrap}.kg-mini{display:grid;grid-template-columns:1fr 1fr;gap:7px;margin-bottom:16px}.kg-mini div{padding:9px;border-radius:6px;background:#edf1f5}.kg-mini strong{display:block;font-size:15px}.kg-mini span{font-size:10px;color:#7b8794}.kg-relation{padding:9px;border-left:3px solid #ef4444;background:#f8fafb;font-size:11px;line-height:1.55;color:#5f6b77;white-space:pre-wrap}.kg-empty{font-size:12px;line-height:1.7;color:#7b8794}
@media(max-width:900px){.kg-body{grid-template-columns:150px minmax(0,1fr) 205px}.kg-side{padding:12px 9px}.kg-sub{max-width:220px}}
@media(max-width:700px){.kg-body{grid-template-columns:150px minmax(0,1fr)}.kg-side.right{display:none}.kg-sub{display:none}}
</style>
"""


UI_SCRIPT = r"""
<script>
(() => {
  const config = window.KG_CONFIG;
  const app = document.getElementById("kg-app");
  const originalNodes = nodes.get().map(item => ({...item}));
  const originalEdges = edges.get().map(item => ({...item}));
  const nodeById = new Map(originalNodes.map(item => [String(item.id), item]));
  const enabledTypes = new Set(config.types);
  let currentMode = config.defaultMode;
  let selectedId = config.defaultNode || null;

  const status = document.getElementById("kg-status");
  const detailType = document.getElementById("kg-detail-type");
  const detailName = document.getElementById("kg-detail-name");
  const detailDescription = document.getElementById("kg-detail-description");
  const detailDegree = document.getElementById("kg-detail-degree");
  const detailEvidence = document.getElementById("kg-detail-evidence");
  const detailRelation = document.getElementById("kg-detail-relation");

  function updateStatus(visibleCount) {
    if (config.mode === "query") {
      status.textContent = `本次命中 ${visibleCount} 个实体`;
    } else if (currentMode === "global") {
      status.textContent = `全部节点 · 已显示 ${visibleCount} 个实体`;
    } else {
      status.textContent = `当前实体及其直接关系 · ${visibleCount} 个实体`;
    }
  }

  function visibleNodeIds() {
    const allowed = new Set(
      originalNodes
        .filter(node => enabledTypes.has(node.entityType || "other"))
        .map(node => String(node.id))
    );
    if (config.mode !== "full" || currentMode === "global") {
      return allowed;
    }
    if (!selectedId) return allowed;
    const focused = new Set([String(selectedId)]);
    network.getConnectedNodes(selectedId).forEach(id => focused.add(String(id)));
    return new Set([...allowed].filter(id => focused.has(id)));
  }

  function applyVisibility({fit = true} = {}) {
    const visible = visibleNodeIds();
    nodes.update(originalNodes.map(node => ({
      id: node.id,
      hidden: !visible.has(String(node.id)),
    })));
    edges.update(originalEdges.map(edge => ({
      id: edge.id,
      hidden: !visible.has(String(edge.from)) || !visible.has(String(edge.to)),
    })));
    updateStatus(visible.size);
    if (fit && visible.size) {
      network.fit({nodes: [...visible], animation: false});
    }
  }

  function showNode(nodeId) {
    const node = nodeById.get(String(nodeId));
    if (!node) return;
    selectedId = String(nodeId);
    detailType.style.setProperty("--detail-color", node.color?.background || "#9aa3ad");
    detailType.querySelector("span").textContent = node.typeLabel || "其他";
    detailName.textContent = node.displayLabel || String(node.id);
    detailDescription.textContent = node.description || "暂无实体说明。";
    detailDegree.textContent = String(node.degree || 0);
    detailEvidence.textContent = String(node.evidenceCount || 0);
    detailRelation.textContent = "点击关系线可以查看关系类型、说明和证据。";
    if (!node.isCritical) nodes.update({id: node.id, label: node.displayLabel});
  }

  function showEdge(edgeId) {
    const edge = edges.get(edgeId);
    if (!edge) return;
    detailType.style.setProperty("--detail-color", "#ef4444");
    detailType.querySelector("span").textContent = "关系";
    const source = nodeById.get(String(edge.from))?.displayLabel || edge.from;
    const target = nodeById.get(String(edge.to))?.displayLabel || edge.to;
    detailName.textContent = `${source} → ${target}`;
    detailDescription.textContent = edge.description || "暂无关系说明。";
    detailDegree.textContent = edge.relationLabel || "其他关系";
    detailEvidence.textContent = String(edge.evidenceCount || 0);
    detailRelation.textContent = edge.evidenceText || "暂无可展示的原文证据。";
  }

  function setMode(mode) {
    currentMode = mode;
    app.querySelectorAll("[data-kg-mode]").forEach(button => {
      button.classList.toggle("active", button.dataset.kgMode === mode);
    });
    applyVisibility();
  }

  function findNode() {
    const query = document.getElementById("kg-search-input").value.trim().toLocaleLowerCase();
    if (!query) return;
    const match = originalNodes.find(node =>
      String(node.displayLabel || node.id).toLocaleLowerCase() === query
    ) || originalNodes.find(node =>
      String(node.displayLabel || node.id).toLocaleLowerCase().includes(query)
    );
    if (!match) {
      status.textContent = `未找到实体：${query}`;
      return;
    }
    selectedId = String(match.id);
    if (config.mode === "full") setMode("focus");
    network.selectNodes([match.id]);
    network.focus(match.id, {scale: 1.25, animation: {duration: 260}});
    showNode(match.id);
  }

  app.querySelectorAll(".kg-filter").forEach(button => {
    button.addEventListener("click", () => {
      const type = button.dataset.type;
      enabledTypes.has(type) ? enabledTypes.delete(type) : enabledTypes.add(type);
      button.classList.toggle("off", !enabledTypes.has(type));
      applyVisibility();
    });
  });
  app.querySelectorAll("[data-kg-mode]").forEach(button => {
    button.addEventListener("click", () => setMode(button.dataset.kgMode));
  });
  document.getElementById("kg-search-button").addEventListener("click", findNode);
  document.getElementById("kg-search-input").addEventListener("keydown", event => {
    if (event.key === "Enter") findNode();
  });
  document.getElementById("kg-reset").addEventListener("click", () => {
    enabledTypes.clear();
    config.types.forEach(type => enabledTypes.add(type));
    app.querySelectorAll(".kg-filter").forEach(button => button.classList.remove("off"));
    selectedId = config.defaultNode || null;
    currentMode = config.defaultMode;
    app.querySelectorAll("[data-kg-mode]").forEach(button => {
      button.classList.toggle("active", button.dataset.kgMode === currentMode);
    });
    network.unselectAll();
    applyVisibility();
    if (selectedId) showNode(selectedId);
  });
  document.getElementById("kg-zoom-in").addEventListener("click", () => {
    network.moveTo({scale: Math.min(network.getScale() * 1.2, 3), animation: false});
  });
  document.getElementById("kg-zoom-out").addEventListener("click", () => {
    network.moveTo({scale: Math.max(network.getScale() / 1.2, 0.03), animation: false});
  });
  document.getElementById("kg-center").addEventListener("click", () => {
    if (selectedId) network.focus(selectedId, {scale: 1.1, animation: {duration: 220}});
    else network.fit({animation: false});
  });
  document.getElementById("kg-export").addEventListener("click", () => {
    network.redraw();
    const canvas = document.querySelector("#mynetwork canvas");
    if (!canvas) return;
    const link = document.createElement("a");
    link.download = "medical-knowledge-graph.png";
    link.href = canvas.toDataURL("image/png");
    link.click();
  });

  network.on("selectNode", params => params.nodes[0] && showNode(params.nodes[0]));
  network.on("selectEdge", params => params.edges[0] && showEdge(params.edges[0]));
  network.on("hoverNode", params => {
    const node = nodeById.get(String(params.node));
    if (node && !node.isCritical) nodes.update({id: node.id, label: node.displayLabel});
  });
  network.on("blurNode", params => {
    const node = nodeById.get(String(params.node));
    if (node && !node.isCritical && String(params.node) !== selectedId) {
      nodes.update({id: node.id, label: ""});
    }
  });

  setTimeout(() => {
    applyVisibility({fit: true});
    if (selectedId && nodeById.has(selectedId)) showNode(selectedId);
  }, 60);
})();
</script>
"""


def _filters_html(type_counts: Counter[str]) -> str:
    buttons = []
    for entity_type in _TYPE_ORDER:
        count = type_counts.get(entity_type, 0)
        if not count:
            continue
        buttons.append(
            '<button class="kg-filter" data-type="{entity_type}" '
            'style="--type-color:{color}">'
            '<i class="kg-dot"></i><span class="name">{label}</span>'
            '<span class="count">{count:,}</span></button>'.format(
                entity_type=entity_type,
                color=TYPE_COLORS[entity_type],
                label=TYPE_LABELS[entity_type],
                count=count,
            )
        )
    return "".join(buttons)


def _shell_html(
    *,
    title: str,
    subtitle: str,
    type_counts: Counter[str],
    mode: str,
) -> str:
    if mode == "full":
        mode_control = (
            '<div class="kg-segment">'
            '<button data-kg-mode="focus">聚焦当前</button>'
            '<button class="active" data-kg-mode="global">全部节点</button>'
            "</div>"
        )
    else:
        mode_control = '<div class="kg-mode-label">本次命中</div>'
    return f"""
<div id="kg-app">
  <header class="kg-head">
    <div class="kg-mark">十</div>
    <div><div class="kg-title">{html.escape(title)}</div><div class="kg-sub">{html.escape(subtitle)}</div></div>
    <div class="kg-grow"></div>
    <button id="kg-export" class="kg-btn">导出图片</button>
  </header>
  <div class="kg-toolbar">
    <div class="kg-search"><input id="kg-search-input" placeholder="搜索疾病、症状、药物或检查"><button id="kg-search-button">定位</button></div>
    {mode_control}<div class="kg-grow"></div>
    <button id="kg-reset" class="kg-btn">重置视图</button>
  </div>
  <div class="kg-body">
    <aside class="kg-side left"><div class="kg-section-title">实体类型<span>点击筛选</span></div>{_filters_html(type_counts)}</aside>
    <main class="kg-main"><div id="mynetwork"></div><div id="kg-status" class="kg-status"></div>
      <div class="kg-legend">拖动画布 · 滚轮缩放 · 点击节点或关系查看详情</div>
      <div class="kg-controls"><button id="kg-zoom-in" aria-label="放大">＋</button><button id="kg-zoom-out" aria-label="缩小">－</button><button id="kg-center" aria-label="居中">◎</button></div>
    </main>
    <aside class="kg-side right"><div id="kg-detail-type" class="kg-type" style="--detail-color:#9aa3ad"><i></i><span>实体</span></div>
      <div id="kg-detail-name" class="kg-name">点击节点查看详情</div><div id="kg-detail-description" class="kg-desc">可查看实体类型、关联数量和证据情况。</div>
      <div class="kg-mini"><div><strong id="kg-detail-degree">0</strong><span>直接关系</span></div><div><strong id="kg-detail-evidence">0</strong><span>证据数量</span></div></div>
      <div class="kg-section-title">关系与证据</div><div id="kg-detail-relation" class="kg-relation">点击关系线查看关系说明。</div>
    </aside>
  </div>
</div>
"""


def _network_options() -> dict[str, Any]:
    return {
        "layout": {"improvedLayout": False},
        "interaction": {
            "hover": True,
            "keyboard": True,
            "multiselect": False,
            "navigationButtons": False,
            "hideEdgesOnDrag": True,
            "tooltipDelay": 160,
        },
        "physics": {"enabled": False},
        "nodes": {
            "shape": "dot",
            "borderWidth": 1,
            "font": {
                "face": "Microsoft YaHei, Arial",
                "size": 13,
                "color": "#20272E",
                "strokeWidth": 4,
                "strokeColor": "#F7F9FB",
            },
        },
        "edges": {
            "width": 0.7,
            "hoverWidth": 1.4,
            "selectionWidth": 2,
            "smooth": False,
            "arrowStrikethrough": False,
            "color": {
                "color": "rgba(133,145,157,0.22)",
                "highlight": "#EF4444",
                "hover": "#F07B26",
                "inherit": False,
            },
        },
    }


def _render_html(
    nodes_data: list[dict[str, Any]],
    edges_data: list[dict[str, Any]],
    *,
    title: str,
    subtitle: str,
    mode: str,
    default_node: str | None,
) -> str:
    network = Network(
        height="100%",
        width="100%",
        directed=True,
        bgcolor="#F7F9FB",
        font_color="#20272E",
        cdn_resources="in_line",
    )
    for node in nodes_data:
        network.add_node(
            node["id"],
            label=node["display_label"] if node["is_critical"] else "",
            title=node["display_label"],
            x=node["x"],
            y=node["y"],
            physics=False,
            size=node["size"],
            color={
                "background": TYPE_COLORS[node["entity_type"]],
                "border": TYPE_COLORS[node["entity_type"]],
                "highlight": {"background": "#EF4444", "border": "#20272E"},
                "hover": {"background": TYPE_COLORS[node["entity_type"]], "border": "#20272E"},
            },
            displayLabel=node["display_label"],
            entityType=node["entity_type"],
            typeLabel=TYPE_LABELS[node["entity_type"]],
            description=node["description"],
            degree=node["degree"],
            evidenceCount=node["evidence_count"],
            isCritical=node["is_critical"],
        )
    for edge in edges_data:
        network.add_edge(
            edge["source"],
            edge["target"],
            arrows={"to": {"enabled": True, "scaleFactor": 0.2}},
            relationLabel=edge["relation_label"],
            description=edge["description"],
            evidenceCount=edge["evidence_count"],
            evidenceText=edge["evidence_text"],
        )
    network.set_options(json.dumps(_network_options(), ensure_ascii=False))
    generated = network.generate_html()
    type_counts = Counter(node["entity_type"] for node in nodes_data)
    shell = _shell_html(title=title, subtitle=subtitle, type_counts=type_counts, mode=mode)
    card_pattern = re.compile(
        r'<div class="card" style="width: 100%">.*?'
        r'<div id="mynetwork" class="card-body"></div>\s*</div>',
        re.DOTALL,
    )
    generated, replacements = card_pattern.subn(shell, generated, count=1)
    if replacements != 1:
        raise RuntimeError("Unable to locate the PyVis graph container")
    config = {
        "mode": mode,
        "defaultMode": "global" if mode == "full" else "query",
        "defaultNode": default_node,
        "types": sorted(type_counts),
    }
    config_script = (
        "<script>window.KG_CONFIG="
        + json.dumps(config, ensure_ascii=False).replace("</", "<\\/")
        + ";</script>"
    )
    generated = generated.replace("</head>", UI_CSS + "</head>")
    return generated.replace("</body>", config_script + UI_SCRIPT + "</body>")


def build_full_graph_html(
    graph: nx.Graph,
    *,
    title: str = "医疗知识图谱",
    label_count: int = 80,
    max_nodes: int = 0,
    min_degree: int = 0,
) -> str:
    """Build the all-node graph page with a static, deterministic layout."""
    if label_count < 1 or max_nodes < 0 or min_degree < 0:
        raise ValueError("graph display limits must be non-negative")

    node_ids = [
        str(node_id)
        for node_id, degree in graph.degree()
        if degree >= min_degree
    ]
    if max_nodes and len(node_ids) > max_nodes:
        node_ids = sorted(
            node_ids,
            key=lambda node_id: (-graph.degree(node_id), node_id),
        )[:max_nodes]
    selected = set(node_ids)
    display_graph = graph.subgraph(node_ids)
    positions = _full_layout(node_ids, graph)
    critical = set(
        sorted(
            node_ids,
            key=lambda node_id: (-graph.degree(node_id), node_id),
        )[:label_count]
    )

    nodes_data = []
    for node_id in node_ids:
        data = dict(graph.nodes[node_id])
        entity_type = normalize_entity_type(data.get("entity_type"))
        degree = display_graph.degree(node_id)
        x, y = positions[node_id]
        nodes_data.append(
            {
                "id": node_id,
                "display_label": str(data.get("entity_id") or node_id),
                "entity_type": entity_type,
                "description": compact_text(data.get("description")) or "暂无实体说明。",
                "degree": degree,
                "evidence_count": _source_count(data.get("source_id")),
                "is_critical": node_id in critical,
                "size": min(28, 6 + math.log2(max(degree, 1) + 1) * 2.6),
                "x": x,
                "y": y,
            }
        )

    edges_data = []
    if graph.is_multigraph():
        edge_iterator = graph.edges(data=True, keys=True)
        normalized_edges = (
            (source, target, data) for source, target, _key, data in edge_iterator
        )
    else:
        normalized_edges = graph.edges(data=True)
    for source, target, data in normalized_edges:
        source = str(source)
        target = str(target)
        if source not in selected or target not in selected:
            continue
        edges_data.append(
            {
                "source": source,
                "target": target,
                "relation_label": relation_label(data.get("keywords")),
                "description": compact_text(data.get("description")) or "暂无关系说明。",
                "evidence_count": _source_count(data.get("source_id")),
                "evidence_text": "来源证据可在问答页面的原文标签中查看。",
            }
        )

    default_node = None
    subtitle = "可查看全部节点，也可搜索实体后聚焦其直接关系"
    return _render_html(
        nodes_data,
        edges_data,
        title=title,
        subtitle=subtitle,
        mode="full",
        default_node=default_node,
    )


def build_query_graph_html(result: dict[str, Any]) -> str:
    """Build the interactive graph shown after a medical QA request."""
    subgraph = result.get("subgraph") or {}
    raw_nodes = subgraph.get("nodes") or []
    raw_edges = subgraph.get("edges") or []
    positions = _query_layout(raw_nodes, raw_edges)
    degree = Counter()
    for edge in raw_edges:
        degree[str(edge.get("source"))] += 1
        degree[str(edge.get("target"))] += 1

    graph_evidence = result.get("graph_evidence") or []
    evidence_by_edge: dict[tuple[str, str, str], dict[str, Any]] = {}
    for item in graph_evidence:
        key = (
            str(item.get("source")),
            str(item.get("target")),
            str(item.get("relation")),
        )
        evidence_by_edge[key] = item

    nodes_data = []
    for node in raw_nodes:
        node_id = str(node.get("id") or "").strip()
        if not node_id:
            continue
        entity_type = normalize_entity_type(node.get("entity_type"))
        x, y = positions.get(node_id, (0, 0))
        related_evidence = sum(
            1
            for item in graph_evidence
            if str(item.get("source")) == node_id or str(item.get("target")) == node_id
        )
        nodes_data.append(
            {
                "id": node_id,
                "display_label": str(node.get("label") or node_id),
                "entity_type": entity_type,
                "description": compact_text(node.get("description")) or "本次问答命中的医疗实体。",
                "degree": degree[node_id],
                "evidence_count": related_evidence,
                "is_critical": True,
                "size": 23 if degree[node_id] == max(degree.values(), default=0) else 15,
                "x": x,
                "y": y,
            }
        )

    edges_data = []
    for edge in raw_edges:
        source = str(edge.get("source"))
        target = str(edge.get("target"))
        relation = str(edge.get("relation") or "related_to")
        evidence = evidence_by_edge.get((source, target, relation), {})
        edges_data.append(
            {
                "source": source,
                "target": target,
                "relation_label": str(
                    evidence.get("relation_label") or relation_label(relation)
                ),
                "description": compact_text(
                    evidence.get("description") or edge.get("description")
                )
                or "暂无关系说明。",
                "evidence_count": 1 if evidence else 0,
                "evidence_text": compact_text(evidence.get("source_excerpt"), 520)
                or "暂无可展示的原文证据。",
            }
        )

    default_node = None
    if graph_evidence:
        default_node = str(graph_evidence[0].get("source") or "") or None
    if not default_node and nodes_data:
        default_node = nodes_data[0]["id"]
    question = str(result.get("question") or "").strip()
    subtitle = f"问题：{question}" if question else "本次问答命中的相关节点与关系"
    return _render_html(
        nodes_data,
        edges_data,
        title="回答命中图谱",
        subtitle=subtitle,
        mode="query",
        default_node=default_node,
    )
