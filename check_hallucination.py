# -*- coding: utf-8 -*-
"""图谱幻觉体检：校验每条关系的两端是否真的出现在来源 chunk 原文里"""
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

WORKING_DIR = Path(os.getenv("MEDICAL_WORKING_DIR", "medical_kg"))
GRAPH = WORKING_DIR / "graph_chunk_entity_relation.graphml"
CHUNKS = WORKING_DIR / "kv_store_text_chunks.json"
REPORT = WORKING_DIR / "hallucination_report.json"

NS = {"g": "http://graphml.graphdrawing.org/xmlns"}

with CHUNKS.open(encoding="utf-8") as stream:
    chunks = json.load(stream)
print(f"载入 chunk: {len(chunks)} 个")

root = ET.parse(GRAPH).getroot()
keymap = {k.get("id"): k.get("attr.name") for k in root.findall("g:key", NS)}

def get_data(elem, name):
    for d in elem.findall("g:data", NS):
        if keymap.get(d.get("key")) == name:
            return (d.text or "")
    return ""

edges = root.findall(".//g:edge", NS)
print(f"图谱边: {len(edges)} 条\n")

hallucinated, ok, no_src = [], 0, 0
for e in edges:
    src = (e.get("source") or "").strip()
    tgt = (e.get("target") or "").strip()
    sid = get_data(e, "source_id")
    if not sid:
        no_src += 1
        continue
    cids = [c.strip() for c in sid.split("<SEP>") if c.strip()]
    hit = False
    for cid in cids:
        text = (chunks.get(cid) or {}).get("content", "")
        if not text:
            continue
        if src in text and tgt in text:
            hit = True
            break
    if hit:
        ok += 1
    else:
        hallucinated.append({"src": src, "tgt": tgt, "chunks": cids})

# 关系类型被当成实体
BAD = {"treats", "has_symptom", "causes", "complication", "exam", "contraindicates"}
bad_type_edges = [{"src": e.get("source"), "tgt": e.get("target")}
                  for e in edges
                  if (e.get("source") in BAD or e.get("target") in BAD)]

rate = len(hallucinated) / max(len(edges), 1) * 100

print("===== 体检结果 =====")
print(f"  两端都在原文里 : {ok}")
print(f"  找不到出处     : {len(hallucinated)}")
print(f"  无 source_id   : {no_src}")
print(f"  疑似幻觉率     : {rate:.1f}%")
print(f"  关系类型当实体 : {len(bad_type_edges)} 条")

print("\n疑似幻觉样例（前 30 条）：")
for x in hallucinated[:30]:
    print(f"  {x['src']}  ->  {x['tgt']}")

print("\n关系类型被当实体（前 20 条）：")
for x in bad_type_edges[:20]:
    print(f"  {x['src']}  ->  {x['tgt']}")

with REPORT.open("w", encoding="utf-8") as stream:
    json.dump(
        {
            "rate_pct": round(rate, 1),
            "hallucinated": hallucinated,
            "bad_type_edges": bad_type_edges,
        },
        stream,
        ensure_ascii=False,
        indent=2,
    )
print(f"\n报告已存 {REPORT.resolve()}")
