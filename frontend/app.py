"""Streamlit frontend for the AI Data Science Team.

Talks to the FastAPI backend (``DS_TEAM_API_BASE`` env, default
``http://localhost:8000``) and walks the user through: upload -> run pipeline
-> inspect EDA / models / report -> chat with the dataset.
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
import ui

API_BASE = os.environ.get("DS_TEAM_API_BASE", "http://localhost:8000").rstrip("/")
API_KEY = os.environ.get("DS_TEAM_API_KEY", "").strip()
ASSETS_DIR = Path(__file__).resolve().parents[1] / "assets"


def _auth_headers() -> dict[str, str]:
    """Send the API key when the backend has auth enabled."""
    return {"X-API-Key": API_KEY} if API_KEY else {}


STATUS_LABELS = {
    "pending": "Queued",
    "running": "Running",
    "succeeded": "Complete",
    "failed": "Stopped",
}
STATUS_TONES = {
    "pending": "neutral",
    "running": "warning",
    "succeeded": "success",
    "failed": "danger",
}
RATE_METRICS = {"accuracy", "f1", "roc_auc"}
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "f1": "F1 score",
    "roc_auc": "ROC AUC",
    "r2": "R squared",
    "mse": "MSE",
    "rmse": "RMSE",
    "mae": "MAE",
}

ui.setup("AI Data Science Team", icon=ui.app_icon(ASSETS_DIR))


def _api(base: str) -> str:
    return base.rstrip("/")


def _provider_label(provider: str) -> str:
    return {
        "mock": "Local demo model",
        "openai": "OpenAI",
        "anthropic": "Anthropic",
        "groq": "Groq",
        "ollama": "Ollama",
    }.get(provider.lower(), provider.replace("_", " ").title())


def _metric_label(key: str) -> str:
    return METRIC_LABELS.get(key, key.replace("_", " ").title())


def _metric_value(key: str, value: object) -> str:
    if value is None:
        return "Not reported"
    if not isinstance(value, (int, float)):
        return str(value)
    if key in RATE_METRICS:
        return f"{value * 100:.1f}%"
    return f"{value:,.3f}"


def _source_label(plot: dict) -> str:
    column = str(plot.get("column") or "").replace("_", " ").strip()
    kind = {
        "histogram": "Distribution",
        "bar": "Counts",
        "boxplot": "By target",
        "correlation": "Correlation",
    }.get(str(plot.get("type", "")), "Figure")
    return f"{kind}: {column}" if column else kind


def main() -> None:
    base = _api(API_BASE)
    provider = _probe(base)
    online = bool(provider)

    with st.sidebar:
        ui.sidebar_brand("AI Data Science Team", "Autonomous analysis")
        st.markdown(
            ui.badge(
                "Service online" if online else "Service offline",
                "success" if online else "danger",
            ),
            unsafe_allow_html=True,
        )
        st.caption(
            f"Reasoning model: {_provider_label(provider)}"
            if online
            else "The analysis service is not responding right now."
        )
        st.divider()
        st.caption("Pipeline")
        st.markdown(
            "- Plan the approach\n"
            "- Profile the dataset\n"
            "- Clean and impute\n"
            "- Engineer features\n"
            "- Train and compare models\n"
            "- Evaluate and explain\n"
            "- Write the report"
        )
        st.divider()
        st.caption(
            "Upload a CSV, choose a target if you have one, and the team runs "
            "the full analysis end to end."
        )

    ui.page_header(
        "AI data science team",
        "Upload a dataset and let a multi-agent pipeline profile, clean, model "
        "and report on it.",
        eyebrow="AUTONOMOUS ANALYSIS",
        badge="Service online" if online else "Service offline",
        badge_tone="success" if online else "danger",
        icon_name="database",
    )

    tab_run, tab_history, tab_chat = st.tabs(["Run analysis", "History", "Chat"])

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
    ui.section(
        "Dataset",
        "Start with a CSV. The planner picks a target when you leave the field blank.",
    )

    uploaded = st.file_uploader("CSV file", type=["csv"], label_visibility="collapsed")
    st.caption("CSV only, up to 200 MB. The first row should contain column names.")

    target = st.text_input(
        "Target column",
        placeholder="Leave blank to auto-detect",
        help="The column the models should predict.",
    )
    tune = st.checkbox(
        "Tune hyperparameters",
        value=False,
        help="Runs a wider search. Slower, but can improve the best model.",
    )
    start = st.button("Run pipeline", type="primary", width="stretch")

    if start:
        if uploaded is None:
            ui.error_state(
                "Add a CSV file before running the pipeline.",
                title="No dataset selected",
                hint="Choose a file in the uploader above.",
            )
        else:
            run_analysis(base, uploaded, target, tune)

    result = st.session_state.get("result")
    if result:
        render_result(result)
    elif uploaded is None:
        ui.empty_state(
            "Ready when you are",
            "Upload a dataset to start the analysis. Nothing leaves this "
            "session until you run the pipeline.",
            hint="A small CSV finishes in a couple of minutes.",
            icon_name="upload",
        )


def run_analysis(base: str, uploaded, target: str | None, tune: bool) -> None:
    st.session_state.pop("result", None)
    with st.spinner("Uploading dataset..."):
        try:
            upload_resp = requests.post(
                f"{base}/upload",
                files={"file": (uploaded.name, uploaded.getvalue(), "text/csv")},
                headers=_auth_headers(),
                timeout=60,
            )
        except requests.RequestException:
            ui.error_state(
                "The analysis service could not be reached.",
                title="Upload failed",
                hint="Check the connection and try again.",
            )
            return
    if upload_resp.status_code == 401:
        ui.error_state(
            "The analysis service rejected this session's API key.",
            title="Not authorized",
            hint="Set DS_TEAM_API_KEY to the same value as the backend API_KEY.",
        )
        return
    if upload_resp.status_code != 200:
        ui.error_state(
            "The service could not read this file. Check that it is a valid CSV "
            "with a header row.",
            title="Upload failed",
        )
        return
    filename = upload_resp.json()["filename"]

    try:
        run_resp = requests.post(
            f"{base}/workflow",
            json={"filename": filename, "target": target or None, "tune": tune},
            headers=_auth_headers(),
            timeout=30,
        )
    except requests.RequestException:
        ui.error_state(
            "The analysis service stopped responding before the run started.",
            title="Could not start the pipeline",
            hint="Try again in a moment.",
        )
        return
    if run_resp.status_code != 200:
        ui.error_state(
            "The service did not accept this analysis request. Try a smaller "
            "dataset or clear the target column.",
            title="Could not start the pipeline",
        )
        return
    run_id = run_resp.json()["run_id"]

    st.session_state["run_id"] = run_id
    result = poll_workflow(base, run_id)
    st.session_state["result"] = result
    if result:
        st.session_state["report_markdown"] = result.get("report_markdown", "")


def poll_workflow(base: str, run_id: str) -> dict | None:
    progress = st.progress(0.0)
    status_box = st.empty()
    last = "pending"
    status_box.caption("Queued. The team starts as soon as a worker is free.")

    for _ in range(600):  # up to ~10 minutes
        try:
            record = requests.get(
                f"{base}/workflow/{run_id}", headers=_auth_headers(), timeout=30
            ).json()
        except (requests.RequestException, ValueError):
            time.sleep(1)
            continue

        status = record.get("status", "pending")
        if status != last:
            label = STATUS_LABELS.get(status, status.title())
            status_box.caption(f"{label}. The page updates as each agent finishes.")
            last = status
        if status == "succeeded":
            progress.progress(1.0)
            status_box.caption("Analysis complete.")
            return record.get("result")
        if status == "failed":
            progress.progress(1.0)
            status_box.empty()
            ui.error_state(
                "The pipeline stopped before finishing. Check the dataset and "
                "try again.",
                title="Analysis stopped",
                hint="A cleaner file with consistent column types usually helps.",
            )
            return None

        progress.progress(0.35 if status == "running" else 0.08)

    status_box.empty()
    ui.error_state(
        "The analysis is taking longer than expected.",
        title="Still running",
        hint="Open History in a moment to load the finished run.",
    )
    return None


def render_result(result: dict) -> None:
    st.divider()
    ui.section(
        "Analysis",
        "The plan, the strongest model and the report produced for this dataset.",
    )

    plan = result.get("plan")
    if plan:
        with st.expander("Pipeline plan", expanded=False):
            st.markdown(plan)

    plots = [
        plot
        for plot in (result.get("plots") or [])
        if Path(str(plot.get("path", ""))).is_file()
    ]
    if plots:
        with st.expander(f"EDA figures ({len(plots)})", expanded=True):
            columns = st.columns(2)
            for index, plot in enumerate(plots):
                with columns[index % 2]:
                    st.image(
                        plot["path"],
                        caption=_source_label(plot),
                        width="stretch",
                    )

    best = result.get("best_model") or {}
    metrics = best.get("test_metrics") or {}
    model_name = best.get("model") or "Best model"
    if metrics:
        ui.section("Best model", str(model_name))
        ui.metric_grid(
            [
                {"label": _metric_label(name), "value": _metric_value(name, value)}
                for name, value in metrics.items()
            ]
        )
    elif model_name != "Best model":
        ui.metric_grid([{"label": "Selected model", "value": str(model_name)}])

    models = result.get("training_summary", {}).get("results") or []
    if models:
        with st.expander("Model leaderboard"):
            st.dataframe(
                _leaderboard_table(models),
                width="stretch",
                hide_index=True,
            )

    importance = result.get("evaluation_summary", {}).get("feature_importance") or {}
    if importance:
        with st.expander("Feature importance", expanded=True):
            rows = sorted(
                (
                    {"Feature": str(name).replace("_", " "), "Importance": value}
                    for name, value in importance.items()
                ),
                key=lambda row: row["Importance"],
                reverse=True,
            )[:12]
            st.bar_chart(
                pd.DataFrame(rows),
                x="Feature",
                y="Importance",
                horizontal=True,
                color=ui.PRIMARY,
            )

    report = result.get("report_markdown")
    if report:
        with st.expander("Technical and business report", expanded=True):
            st.markdown(_clean_report(report))

    st.session_state["report_markdown"] = report or ""


def _leaderboard_table(models: list[dict]) -> list[dict]:
    rows = []
    for entry in models:
        if "error" in entry:
            continue
        row = {
            "Model": entry.get("model", "Model"),
            "CV mean": _round(entry.get("cv_mean")),
            "CV std": _round(entry.get("cv_std")),
        }
        for key, value in (entry.get("test_metrics") or {}).items():
            row[_metric_label(key)] = _round(value)
        rows.append(row)
    return rows


def _round(value: object) -> object:
    return round(value, 4) if isinstance(value, float) else value


_REPORT_PATH = re.compile(r"`(?:[A-Za-z]:\\|/)[^`]*[\\/]([^`\\/]+)`")


def _clean_report(markdown: str) -> str:
    """Show file names instead of server paths that mean nothing in a browser."""
    return _REPORT_PATH.sub(lambda match: f"`{match.group(1)}`", markdown)


# --------------------------------------------------------------------------- #
# History tab
# --------------------------------------------------------------------------- #
def history_tab(base: str) -> None:
    try:
        runs = requests.get(f"{base}/runs", headers=_auth_headers(), timeout=15).json()
    except (requests.RequestException, ValueError):
        ui.error_state(
            "The analysis service is not responding right now.",
            title="History unavailable",
            hint="Try again in a moment.",
        )
        return

    if not runs:
        ui.empty_state(
            "No runs yet",
            "Finished analyses appear here so you can reopen their results.",
            hint="Run a pipeline from the first tab to get started.",
            icon_name="clock",
        )
        return

    ui.section(
        "Recent runs",
        "Load a previous analysis to review its report and ask follow-up questions.",
    )

    pending_result = None
    for run in runs[:25]:
        status = str(run.get("status", "pending"))
        created = str(run.get("created_at", ""))[:19].replace("T", " ")
        run_id = str(run.get("run_id", ""))
        st.markdown(
            f'<div style="display:flex;flex-direction:column;gap:2px">'
            f'<div style="display:flex;align-items:center;gap:.65rem;flex-wrap:wrap">'
            f'<span class="ui-brand-name">Run {run_id[:8]}</span>'
            f"{ui.badge(STATUS_LABELS.get(status, status.title()), STATUS_TONES.get(status, 'neutral'))}"
            f"</div>"
            f'<div class="ui-brand-sub">{created}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )
        if st.button("Load analysis", key=f"load_{run_id}"):
            pending_result = _load_run(base, run_id)
        st.divider()

    if pending_result:
        render_result(pending_result)


def _load_run(base: str, run_id: str) -> dict | None:
    try:
        record = requests.get(
            f"{base}/workflow/{run_id}", headers=_auth_headers(), timeout=15
        ).json()
    except (requests.RequestException, ValueError):
        ui.error_state(
            "The run could not be loaded right now.",
            title="Load failed",
            hint="Try again in a moment.",
        )
        return None
    result = record.get("result")
    if not result:
        ui.error_state(
            "This run finished without a result to display.",
            title="Nothing to load",
        )
        return None
    st.session_state["result"] = result
    st.session_state["report_markdown"] = result.get("report_markdown", "")
    return result


# --------------------------------------------------------------------------- #
# Chat tab
# --------------------------------------------------------------------------- #
def chat_tab(base: str) -> None:
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    result = st.session_state.get("result")
    if not result:
        ui.empty_state(
            "No analysis loaded",
            "Run a pipeline or load a previous run, then ask questions about "
            "the dataset and the model.",
            hint="The assistant answers from the analysis context only.",
            icon_name="message",
        )
        return

    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if question := st.chat_input("Ask about the dataset..."):
        st.session_state["messages"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        payload = {"question": question, "context": result}
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = requests.post(
                        f"{base}/chat",
                        json=payload,
                        headers=_auth_headers(),
                        timeout=120,
                    )
                    answer = response.json().get("answer") or (
                        "The assistant did not return an answer. Try rephrasing "
                        "the question."
                    )
                except (requests.RequestException, ValueError):
                    answer = (
                        "The assistant is unavailable right now. Try again in a moment."
                    )
            st.markdown(answer)
        st.session_state["messages"].append({"role": "assistant", "content": answer})


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=30)
def _probe(base: str) -> str | None:
    try:
        data = requests.get(f"{base}/", headers=_auth_headers(), timeout=5).json()
        return data.get("provider")
    except (requests.RequestException, ValueError):
        return None


if __name__ == "__main__":
    main()
