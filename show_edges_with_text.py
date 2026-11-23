# -*- coding: utf-8 -*-
"""把 graphml 每条边 + 对应来源 chunk 原文导出成 HTML 网页"""
import json
import os
import xml.etree.ElementTree as ET
from html import escape
from pathlib import Path

WORKING_DIR = Path(os.getenv("MEDICAL_WORKING_DIR", "medical_kg"))
GRAPH = WORKING_DIR / "graph_chunk_entity_relation.graphml"
CHUNKS = WORKING_DIR / "kv_store_text_chunks.json"
OUT = WORKING_DIR / "edges_view.html"

NS = {"g": "http://graphml.graphdrawing.org/xmlns"}

with CHUNKS.open(encoding="utf-8") as stream:
    chunks = json.load(stream)
root = ET.parse(GRAPH).getroot()
keymap = {k.get("id"): k.get("attr.name") for k in root.findall("g:key", NS)}

def get_data(elem, name):
    for d in elem.findall("g:data", NS):
        if keymap.get(d.get("key")) == name:
            return d.text or ""
    return ""

edges = []
for e in root.findall(".//g:edge", NS):
    src = e.get("source", "")
    tgt = e.get("target", "")
    sid = get_data(e, "source_id")
    desc = get_data(e, "description") or get_data(e, "keywords")
    cids = [c.strip() for c in sid.split("<SEP>") if c.strip()]
    chunks_text = []
    for cid in cids:
        chunk = chunks.get(cid, {})
        content = chunk.get("content", "(原文缺失)")
        chunks_text.append((cid, content))
    edges.append({"src": src, "tgt": tgt, "desc": desc[:300], "chunks": chunks_text})

html = [
    "<html><head><meta charset='utf-8'><title>图谱边 + 原文证据</title>",
    "<style>",
    "body{font-family:'Microsoft YaHei',sans-serif;max-width:900px;margin:20px auto;padding:0 20px;background:#fafafa}",
    "h1{color:#333}",
    ".edge{background:#fff;border:1px solid #ddd;margin:12px 0;padding:14px;border-radius:8px;box-shadow:0 1px 3px rgba(0,0,0,0.05)}",
    ".rel{font-size:18px;font-weight:bold;color:#222;margin-bottom:6px}",
    ".desc{color:#666;font-size:14px;margin:6px 0}",
    ".chunk{background:#f5f5f5;padding:10px;margin:6px 0;border-radius:4px;font-size:14px;white-space:pre-wrap;line-height:1.6}",
    ".cid{color:#999;font-size:12px}",
    "</style></head><body>",
]
html.append(f"<h1>共 {len(edges)} 条关系</h1>")
for i, e in enumerate(edges):
    html.append(f"<div class='edge'>")
    html.append(f"<div class='rel'>G{i+1} · {escape(e['src'])} → {escape(e['tgt'])}</div>")
    if e["desc"]:
        html.append(f"<div class='desc'>关系描述：{escape(e['desc'])}</div>")
    for cid, content in e["chunks"]:
        html.append(
            f"<div class='chunk'><span class='cid'>来源 chunk: {escape(cid)}</span><br>{escape(content)}</div>"
        )
    html.append("</div>")
html.append("</body></html>")

with OUT.open("w", encoding="utf-8") as f:
    f.write("\n".join(html))

print(f"已生成: {OUT}")
print(f"共 {len(edges)} 条边，浏览器打开即可")
