"""Streamlit UI for evidence-grounded medical knowledge-graph QA."""

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from medical_graph_ui import build_query_graph_html
from medical_qa_service import MedicalQAService


WORKING_DIR = os.getenv("MEDICAL_WORKING_DIR", "medical_kg")

ROUTE_LABELS = {
    "graph": "知识图谱",
    "text": "原文检索",
    "hybrid": "混合检索",
    "text_fallback": "图谱未命中，回退原文",
}


@st.cache_resource(show_spinner=False)
def get_qa_service(working_dir: str) -> MedicalQAService:
    """Create one process-wide QA service for all Streamlit reruns."""
    return MedicalQAService(working_dir)


def run_question(question: str, working_dir: str) -> dict:
    return get_qa_service(working_dir).answer(question)


def render_entity_information(result: dict) -> None:
    intent = result.get("intent") or {}
    entities = intent.get("entities") or []
    st.markdown("#### 问题识别结果")
    info_cols = st.columns(3)
    info_cols[0].metric("意图", intent.get("intent_label", "未知"))
    info_cols[1].metric("关系", intent.get("relation") or "无固定关系")
    info_cols[2].metric("识别置信度", f"{float(intent.get('confidence', 0)):.0%}")
    if entities:
        st.dataframe(
            pd.DataFrame(
                [
                    {"实体名称": item.get("name"), "实体类型": item.get("type")}
                    for item in entities
                ]
            ),
            width="stretch",
            hide_index=True,
        )
    else:
        st.info("问题中没有识别到明确实体。")

    graph_items = result.get("graph_evidence") or []
    if graph_items:
        st.markdown("#### 命中的关系")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "证据": item.get("id"),
                        "源实体": item.get("source"),
                        "关系": item.get("relation_label"),
                        "目标实体": item.get("target"),
                        "目标类型": item.get("target_type"),
                        "原文支持": "是"
                        if item.get("direct_text_support")
                        else "否",
                    }
                    for item in graph_items
                ]
            ),
            width="stretch",
            hide_index=True,
        )


def render_original_text(result: dict) -> None:
    displayed = False
    for item in result.get("graph_evidence") or []:
        excerpt = item.get("source_excerpt")
        if not excerpt:
            continue
        displayed = True
        with st.expander(
            f"{item.get('id')} · {item.get('source')} → {item.get('target')}"
        ):
            st.write(excerpt)
            st.caption(f"来源 chunk：{item.get('source_chunk_id') or '未知'}")
    for item in result.get("text_evidence") or []:
        displayed = True
        with st.expander(f"{item.get('id')} · {item.get('chunk_id') or '未知来源'}"):
            st.write(item.get("content") or "")
            st.caption(f"文件来源：{item.get('file_path') or 'unknown_source'}")
    if not displayed:
        st.info("本次回答没有可展示的原始文本证据。")


def render_sidebar() -> str:
    if "page" not in st.session_state:
        st.session_state["page"] = "qa"

    with st.sidebar:
        st.markdown("## 系统导航")
        nav_items = [
            ("智能问答", "qa"),
            ("查看图谱", "graph"),
            ("构建思路", "architecture"),
        ]
        for label, page_key in nav_items:
            button_type = (
                "primary" if st.session_state["page"] == page_key else "secondary"
            )
            if st.button(label, key=f"nav_{page_key}", type=button_type, width="stretch"):
                st.session_state["page"] = page_key
                st.rerun()

        st.divider()
        graph_path = Path(WORKING_DIR) / "graph_chunk_entity_relation.graphml"
        st.caption(f"当前知识库：{WORKING_DIR}")
        if graph_path.is_file():
            st.success("知识图谱已就绪")
        else:
            st.error("知识图谱不可用")
    return st.session_state["page"]


def render_qa_page() -> None:
    st.title("医疗知识图谱智能问答")
    st.caption("基于现有 LightRAG 医疗知识图谱进行意图识别、证据检索和受约束回答。")

    question = st.text_area(
        "请输入医疗问题",
        key="question",
        height=100,
        placeholder="例如：肺炎有哪些症状？",
    )
    ask_col, clear_col = st.columns([1, 5])
    with ask_col:
        submit = st.button("开始问答", type="primary", width="stretch")
    with clear_col:
        if st.button("清空结果"):
            st.session_state.pop("result", None)
            st.rerun()

    if submit:
        graph_path = Path(WORKING_DIR) / "graph_chunk_entity_relation.graphml"
        if not question.strip():
            st.warning("请输入问题。")
        elif not graph_path.is_file():
            st.error(f"未找到知识图谱：{graph_path}")
        else:
            try:
                with st.spinner("正在识别意图并检索知识库……"):
                    st.session_state["result"] = run_question(
                        question.strip(), WORKING_DIR
                    )
            except Exception as exc:
                st.exception(exc)

    result = st.session_state.get("result")
    if not result:
        return

    intent = result.get("intent") or {}
    metrics = st.columns(4)
    metrics[0].metric("识别意图", intent.get("intent_label", "未知"))
    metrics[1].metric(
        "检索方式",
        ROUTE_LABELS.get(result.get("route_used"), result.get("route_used")),
    )
    metrics[2].metric("图谱证据", len(result.get("graph_evidence") or []))
    metrics[3].metric("文本证据", len(result.get("text_evidence") or []))
    timings = result.get("timings") or {}
    if timings.get("total_ms") is not None:
        st.caption(
            "耗时："
            f"检索 {float(timings.get('retrieval_ms', 0)) / 1000:.1f}s · "
            f"生成 {float(timings.get('generation_ms', 0)) / 1000:.1f}s · "
            f"总计 {float(timings.get('total_ms', 0)) / 1000:.1f}s"
        )

    st.markdown("### 回答")
    st.markdown(result.get("answer") or "")
    used_ids = result.get("used_evidence_ids") or []
    if used_ids:
        st.caption(f"使用证据：{', '.join(used_ids)}")
    if result.get("limitations"):
        st.warning(f"证据局限：{result['limitations']}")
    st.info(result.get("disclaimer") or "")

    graph_tab, json_tab, entity_tab, source_tab = st.tabs(
        ["关系图谱", "JSON 明细", "实体信息", "原文"]
    )
    with graph_tab:
        subgraph = result.get("subgraph") or {}
        if subgraph.get("nodes"):
            st.iframe(
                build_query_graph_html(result),
                height=660,
                alt="本次问题命中的医疗知识图谱",
            )
        else:
            st.info("当前回答没有可展示的局部图谱。")
    with json_tab:
        st.download_button(
            "下载 JSON",
            data=json.dumps(result, ensure_ascii=False, indent=2),
            file_name="medical_qa_result.json",
            mime="application/json",
        )
        st.json(result, expanded=False)
    with entity_tab:
        render_entity_information(result)
    with source_tab:
        render_original_text(result)


def render_graph_page() -> None:
    st.title("医疗知识图谱")
    st.caption("浏览当前知识库中的全部实体和关系。问答命中的局部图谱会显示在智能问答结果中。")
    graph_html = Path(WORKING_DIR) / "graph_view.html"
    if not graph_html.is_file():
        st.error(f"未找到图谱页面：{graph_html}")
        st.code("python view_graph.py", language="powershell")
        return
    st.iframe(
        graph_html,
        height=820,
        alt="完整医疗知识图谱",
    )


def render_architecture_page() -> None:
    st.title("项目构建思路")
    architecture_image = Path(__file__).parent / "assets" / "medical_qa_workflow.png"
    if architecture_image.is_file():
        st.image(
            architecture_image,
            caption="LightRAG-MedicalQA 医疗智能问答系统流程",
            width="stretch",
        )
    else:
        st.warning(f"未找到项目流程图：{architecture_image}")
    st.markdown(
        """
        ### 当前系统流程

        1. 用户输入医疗问题
        2. LLM 识别问题意图和医疗实体
        3. 根据意图选择图谱检索、原文检索或混合检索
        4. 对实体类型、关系类型和原文相关性进行过滤
        5. LLM 仅根据检索证据生成带引用的回答
        6. 页面展示答案、关系图谱、JSON 明细、实体信息和原文
        """
    )


st.set_page_config(
    page_title="医疗知识图谱问答",
    page_icon="⚕",
    layout="wide",
)

# Start the process-wide worker once and preload graph/KV/vector storages.
qa_service = get_qa_service(WORKING_DIR)

st.markdown(
    """
    <style>
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #C83E4D !important;
        border-bottom-color: #C83E4D !important;
    }
    [data-testid="stSidebar"] button {
        justify-content: flex-start;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

current_page = render_sidebar()
if current_page == "graph":
    render_graph_page()
elif current_page == "architecture":
    render_architecture_page()
else:
    render_qa_page()
