"""Streamlit frontend for the AI Data Science Team.

Talks to the FastAPI backend (``DS_TEAM_API_BASE`` env, default
``http://localhost:8000``) and walks the user through: upload -> run pipeline
-> inspect EDA / models / report -> chat with the dataset.
"""

from __future__ import annotations

import os
import time

import requests
import streamlit as st

st.set_page_config(page_title="AI Data Science Team", page_icon="📊", layout="wide")

# On Streamlit Cloud there is no localhost backend; set DS_TEAM_API_BASE to
# the URL of the deployed FastAPI service (it stays overridable in the
# sidebar).
API_BASE = os.environ.get("DS_TEAM_API_BASE", "http://localhost:8000")

st.markdown(
    "<style>"
    "body{background:#0E1117}"
    ".stApp{background:#0E1117;color:#E6E6E6}"
    ".block-container{padding-top:2rem;max-width:1400px}"
    "h1,h2,h3{color:#4FC3F7}"
    "div[data-testid='stMetric']{background:#161B22;border:1px solid #30363D;"
    "border-radius:10px;padding:12px}"
    "</style>",
    unsafe_allow_html=True,
)


def _api(base: str) -> str:
    return base.rstrip("/")


def main() -> None:
    base = _api(st.sidebar.text_input("Backend URL", value=API_BASE))
    provider = _probe(base)
    st.sidebar.info(f"LLM provider: `{provider}`" if provider else "Backend not reachable")

    st.title("🤖 AI Data Science Team")
    st.caption(
        "Autonomous multi-agent pipeline: Planner → EDA → Cleaning → Features → "
        "Training → Evaluation → Report → Chat."
    )

    tab_run, tab_history, tab_chat = st.tabs(["🚀 Run Analysis", "🕘 History", "💬 Chat"])

    with tab_run:
        run_tab(base)
    with tab_history:
        history_tab(base)
    with tab_chat:
        chat_tab(base)


# --------------------------------------------------------------------------- #
# Run tab
# --------------------------------------------------------------------------- #
def run_tab(base: str) -> None:
    st.subheader("Upload a dataset")
    uploaded = st.file_uploader("CSV file", type=["csv"], label_visibility="collapsed")

    target = st.text_input(
        "Target column (optional)",
        placeholder="Leave blank to auto-detect",
        help="If blank, the planner picks a sensible default.",
    )

    col1, col2, _ = st.columns([1, 1, 3])
    tune = col1.checkbox("Hyperparameter tuning", value=False)
    start = col2.button("▶ Run full pipeline", type="primary", width="stretch")

    if uploaded is None:
        return
    if start:
        run_analysis(base, uploaded, target, tune)


def run_analysis(base: str, uploaded, target: str | None, tune: bool) -> None:
    with st.spinner("Uploading dataset…"):
        upload_resp = requests.post(
            f"{base}/upload",
            files={"file": (uploaded.name, uploaded.getvalue(), "text/csv")},
            timeout=60,
        )
    if upload_resp.status_code != 200:
        st.error(f"Upload failed: {upload_resp.text}")
        return
    filename = upload_resp.json()["filename"]

    run_resp = requests.post(
        f"{base}/workflow",
        json={"filename": filename, "target": target or None, "tune": tune},
        timeout=30,
    )
    if run_resp.status_code != 200:
        st.error(f"Workflow start failed: {run_resp.text}")
        return
    run_id = run_resp.json()["run_id"]

    st.session_state["run_id"] = run_id
    result = poll_workflow(base, run_id)
    st.session_state["result"] = result
    if result:
        render_result(result)


def poll_workflow(base: str, run_id: str) -> dict | None:
    progress = st.progress(0.0)
    status_box = st.empty()
    steps = ["pending", "running", "succeeded", "failed"]
    last = "pending"

    for _ in range(600):  # up to ~10 minutes
        try:
            record = requests.get(f"{base}/workflow/{run_id}", timeout=30).json()
        except requests.RequestException:
            time.sleep(1)
            continue

        status = record["status"]
        if status != last:
            status_box.info(f"Status: **{status}**")
            last = status
        if status == "succeeded":
            progress.progress(1.0)
            status_box.success("Pipeline finished ✔")
            return record.get("result")
        if status == "failed":
            progress.progress(1.0)
            status_box.error(f"Pipeline failed: {record.get('error')}")
            return None
        progress.progress(min(0.95, steps.index(status) * 0.3 + 0.05))
        time.sleep(1)

    status_box.warning("Timed out waiting for the pipeline.")
    return None


def render_result(result: dict) -> None:
    st.divider()
    st.subheader("📄 Pipeline plan")
    st.markdown(result.get("plan") or "_No plan available._")

    plots = result.get("plots") or []
    if plots:
        with st.expander("📈 EDA plots", expanded=True):
            columns = st.columns(3)
            for i, plot in enumerate(plots):
                with columns[i % 3]:
                    try:
                        st.image(plot["path"], caption=plot.get("column", ""), width="stretch")
                    except FileNotFoundError:
                        st.caption(f"Plot missing: {plot['path']}")

    best = result.get("best_model") or {}
    metrics = best.get("test_metrics") or {}
    cols = st.columns(len(metrics) if metrics else 1)
    for col, (name, value) in zip(cols, metrics.items()):
        col.metric(f"Best {name}", value)

    models = result.get("training_summary", {}).get("results") or []
    if models:
        with st.expander("🧠 Model leaderboard"):
            st.dataframe(_leaderboard_table(models), width="stretch")

    importance = result.get("evaluation_summary", {}).get("feature_importance") or {}
    if importance:
        with st.expander("⭐ Feature importance"):
            st.bar_chart({"importance": importance})

    with st.expander("📝 Technical & business report", expanded=True):
        st.markdown(result.get("report_markdown") or "_No report yet._")

    st.session_state["report_markdown"] = result.get("report_markdown", "")


def _leaderboard_table(models: list[dict]) -> list[dict]:
    rows = []
    for entry in models:
        if "error" in entry:
            continue
        rows.append(
            {
                "model": entry["model"],
                "cv_mean": entry.get("cv_mean"),
                "cv_std": entry.get("cv_std"),
                **{f"test_{k}": v for k, v in (entry.get("test_metrics") or {}).items()},
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# History tab
# --------------------------------------------------------------------------- #
def history_tab(base: str) -> None:
    try:
        runs = requests.get(f"{base}/runs", timeout=15).json()
    except requests.RequestException:
        st.error("Backend not reachable.")
        return
    if not runs:
        st.caption("No runs yet.")
        return

    for run in runs:
        c1, c2, c3, c4 = st.columns([2, 1, 2, 2])
        c1.write(f"`{run['run_id']}`")
        c2.write(run["status"])
        c3.write(run["created_at"][:19])
        if c4.button("Load", key=run["run_id"]):
            record = requests.get(f"{base}/workflow/{run['run_id']}", timeout=15).json()
            st.session_state["result"] = record.get("result")
            st.rerun()


# --------------------------------------------------------------------------- #
# Chat tab
# --------------------------------------------------------------------------- #
def chat_tab(base: str) -> None:
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    result = st.session_state.get("result")
    if not result:
        st.info("Run a pipeline first — the chat answers questions about that analysis.")
        return

    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if question := st.chat_input("Ask about the dataset…"):
        st.session_state["messages"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        payload = {"question": question, "context": result}
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    answer = requests.post(f"{base}/chat", json=payload, timeout=120).json()["answer"]
                except requests.RequestException:
                    answer = "Backend not reachable."
            st.markdown(answer)
        st.session_state["messages"].append({"role": "assistant", "content": answer})


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=30)
def _probe(base: str) -> str | None:
    try:
        data = requests.get(f"{base}/", timeout=5).json()
        return data.get("provider")
    except requests.RequestException:
        return None


if __name__ == "__main__":
    main()