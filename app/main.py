"""FastAPI backend for the AI Data Science Team.

Endpoints
---------
- ``GET  /``              API metadata
- ``GET  /health``        liveness probe
- ``POST /upload``        upload a CSV dataset
- ``POST /workflow``      start a full pipeline run (background)
- ``GET  /workflow/{id}`` poll run status + result
- ``GET  /runs``          list recent runs
- ``POST /chat``          ask questions about an analysed dataset
- ``GET  /reports``       list generated reports
- ``GET  /reports/{name}`` fetch a report's markdown
"""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Any

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    HTTPException,
    Query,
    Security,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field

from app.agents.chatbot import ChatbotAgent
from app.config import get_settings
from app.graph.workflow import agent_names, run_pipeline
from app.services import memory, tracking
from app.services.store import get_store

settings = get_settings()
app = FastAPI(title=settings.app_name, version=settings.app_version)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.backend_origins_list,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require_api_key(x_api_key: str | None = Security(_api_key_header)) -> None:
    """Gate mutating endpoints when ``API_KEY`` is configured (off when empty)."""
    if settings.api_key and x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header.")

_chatbot = ChatbotAgent()


# --------------------------------------------------------------------------- #
# MCP endpoint metadata (optional, requires the `mcp` package)
# --------------------------------------------------------------------------- #
def _mcp_available() -> bool:
    try:
        from app import mcp_server

        return mcp_server._MCP_AVAILABLE
    except Exception:  # noqa: BLE001 - optional integration must never break startup
        return False


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    context: dict[str, Any] = Field(default_factory=dict)


class RunRequest(BaseModel):
    filename: str
    target: str | None = None
    tune: bool = False


class RunResponse(BaseModel):
    run_id: str
    status: str
    detail: str = ""


class SQLRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    dataset_path: str | None = None


class PredictRequest(BaseModel):
    run_id: str
    csv_path: str


class DeployResponse(BaseModel):
    run_id: str
    path: str
    status: str


class PythonRunRequest(BaseModel):
    code: str = Field(..., min_length=1, max_length=20000)
    data_files: list[str] = Field(default_factory=list, max_length=10)
    timeout_seconds: int | None = Field(default=None, ge=1, le=300)


class AnomalyRequest(BaseModel):
    dataset_path: str
    contamination: float = Field(default=0.05, ge=0.005, le=0.5)


class StatsRequest(BaseModel):
    dataset_path: str
    spec: dict[str, Any]


class ForecastRequest(BaseModel):
    dataset_path: str
    column: str | None = None
    horizon: int = Field(default=10, ge=1, le=90)
    period: int | None = Field(default=None, ge=2, le=52)


# --------------------------------------------------------------------------- #
# Lifecycle
# --------------------------------------------------------------------------- #
@app.on_event("startup")
def _ensure_dirs() -> None:
    settings.upload_dir_path.mkdir(parents=True, exist_ok=True)
    settings.model_dir_path.mkdir(parents=True, exist_ok=True)
    settings.report_dir_path.mkdir(parents=True, exist_ok=True)
    settings.deploy_dir_path.mkdir(parents=True, exist_ok=True)
    get_store()


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@app.get("/")
def root() -> dict[str, Any]:
    return {
        "app": settings.app_name,
        "version": settings.app_version,
        "agents": agent_names(),
        "provider": _provider_label(),
        "mcp": _mcp_available(),
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/monitoring/overview")
def monitoring_overview() -> dict[str, Any]:
    """High-level operational overview: run counts and memory size."""
    from app.services.store import get_store

    store = get_store()
    runs = store.list_runs(100)
    statuses: dict[str, int] = {}
    for run in runs:
        statuses[run["status"]] = statuses.get(run["status"], 0) + 1
    return {
        "total_runs": len(runs),
        "by_status": statuses,
        "memory_docs": memory.count(),
        "tracking_mlflow": tracking.available(),
    }


@app.post("/upload", dependencies=[Depends(require_api_key)])
async def upload(file: UploadFile = File(...)) -> dict[str, Any]:  # noqa: B008 - FastAPI injects the file
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    destination = settings.upload_dir_path / Path(file.filename).name
    with destination.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {"filename": destination.name, "path": str(destination)}


@app.post("/workflow", response_model=RunResponse, dependencies=[Depends(require_api_key)])
def start_workflow(req: RunRequest, background_tasks: BackgroundTasks) -> RunResponse:
    dataset_path = settings.upload_dir_path / Path(req.filename).name
    if not dataset_path.is_file():
        raise HTTPException(status_code=404, detail=f"Uploaded file not found: {req.filename}")

    store = get_store()
    run_id = store.create_run(str(dataset_path), req.target)
    background_tasks.add_task(_execute, run_id, req)

    return RunResponse(run_id=run_id, status="pending", detail="Workflow started.")


@app.get("/workflow/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:
    record = get_store().get_run(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
    return record


@app.get("/runs")
def list_runs(limit: int = Query(default=20, ge=1, le=100)) -> list[dict[str, Any]]:
    return get_store().list_runs(limit)


@app.post("/chat", dependencies=[Depends(require_api_key)])
def chat(req: ChatRequest) -> dict[str, str]:
    return {"answer": _chatbot.answer(req.question, req.context)}


@app.get("/reports")
def list_reports() -> list[dict[str, str]]:
    out_dir = settings.report_dir_path
    return [
        {"name": path.name, "path": str(path)}
        for path in sorted(out_dir.glob("report_*.md"), reverse=True)
    ]


@app.get("/reports/{name}")
def get_report(name: str) -> dict[str, str]:
    out_dir = settings.report_dir_path
    candidate = out_dir / Path(name).name
    if not candidate.is_file() or candidate.suffix != ".md":
        raise HTTPException(status_code=404, detail="Report not found.")
    return {"name": candidate.name, "markdown": candidate.read_text(encoding="utf-8")}


@app.post("/sql", dependencies=[Depends(require_api_key)])
def run_sql_query(req: SQLRequest) -> dict[str, Any]:
    """Natural-language SQL question against an uploaded dataset."""
    from app.agents.sql_agent import SQLAgent

    dataset_path = req.dataset_path
    if not dataset_path:
        store = get_store()
        run_id = _latest_run_id(store)
        if run_id:
            record = store.get_run(run_id)
            dataset_path = (record or {}).get("dataset_path") if record else None
    if not dataset_path:
        raise HTTPException(status_code=404, detail="No dataset available; upload or start a run first.")

    dataset = _resolve_upload_dataset(dataset_path)
    return SQLAgent().answer(req.question, str(dataset))


@app.get("/runs/recent")
def recent_runs(limit: int = Query(default=10, ge=1, le=50)) -> list[dict[str, Any]]:
    """Run summaries for dashboards / monitoring."""
    return tracking.recent_runs(limit)


@app.post("/memory/search", dependencies=[Depends(require_api_key)])
def memory_search(query: str, top_k: int = Query(default=3, ge=1, le=10)) -> list[dict[str, Any]]:
    """Search prior run knowledge stored in the vector memory."""
    return memory.search(query, top_k)


@app.post("/deploy/{run_id}", response_model=DeployResponse, dependencies=[Depends(require_api_key)])
def deploy_run(run_id: str) -> DeployResponse:
    """Package a finished run's best model for production serving."""
    from app.agents.deploy import DeploymentAgent

    store = get_store()
    record = store.get_run(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
    result = record.get("result") or {}
    bundle = DeploymentAgent().run({**result, "run_id": run_id})["deployment_summary"]
    if "error" in bundle:
        raise HTTPException(status_code=400, detail=bundle["error"])
    return DeployResponse(run_id=run_id, path=bundle["path"], status="deployed")


@app.post("/predict", dependencies=[Depends(require_api_key)])
def predict(req: PredictRequest) -> dict[str, Any]:
    """Run predictions with a deployed model against a CSV file."""
    bundle_dir = settings.deploy_dir_path / Path(req.run_id).name
    predictor = bundle_dir / "predict.py"
    if not predictor.is_file():
        raise HTTPException(status_code=404, detail="Deployment not found; deploy the run first.")

    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("_bundle_predict", predictor)
    if spec is None or spec.loader is None:
        raise HTTPException(status_code=500, detail="Could not load the predictor module.")
    module = importlib.util.module_from_spec(spec)
    sys.modules["_bundle_predict"] = module
    spec.loader.exec_module(module)

    dataset = settings.upload_dir_path / Path(req.csv_path).name
    if not dataset.is_file():
        raise HTTPException(status_code=404, detail=f"CSV not found: {req.csv_path}")
    try:
        predictions = module.predict_csv(dataset)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc
    return {"run_id": req.run_id, "predictions": predictions}


@app.post("/export/github/{run_id}", dependencies=[Depends(require_api_key)])
def export_github(run_id: str) -> dict[str, Any]:
    """Publish a run's report to a configured GitHub repository."""
    from app.services.github import is_configured, publish_report

    if not is_configured():
        raise HTTPException(status_code=400, detail="GitHub export is not configured.")
    try:
        return publish_report(run_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/python", dependencies=[Depends(require_api_key)])
def run_python_endpoint(req: PythonRunRequest) -> dict[str, Any]:
    """Execute LLM-written analysis code in a sandboxed child interpreter."""
    from app.tools import python_tool

    try:
        scratch = settings.upload_dir_path / f"py_{uuid.uuid4().hex[:12]}"
        sources: dict[str, str] = {}
        for name in req.data_files:
            dataset = _resolve_upload_dataset(name)
            sources[dataset.name] = str(dataset)
        result = python_tool.run_python(
            req.code,
            work_dir=scratch,
            data_files=sources,
            timeout_seconds=req.timeout_seconds or settings.python_tool_timeout,
        )
    except python_tool.DeniedImport as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return result


@app.post("/anomalies", dependencies=[Depends(require_api_key)])
def detect_anomalies_endpoint(req: AnomalyRequest) -> dict[str, Any]:
    """Flag anomalous rows of an uploaded dataset (IsolationForest)."""
    from app.tools.anomaly_tool import detect_anomalies
    from app.tools.csv_tool import load_csv

    dataset = _resolve_upload_dataset(req.dataset_path)
    df = load_csv(str(dataset))
    if len(df) > 200_000:
        df = df.head(200_000)
    try:
        return detect_anomalies(df, contamination=req.contamination)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/stats", dependencies=[Depends(require_api_key)])
def run_statistical_test_endpoint(req: StatsRequest) -> dict[str, Any]:
    """Run a hypothesis test against an uploaded dataset."""
    from app.tools.csv_tool import load_csv
    from app.tools.stats_tool import run_statistical_test

    dataset = _resolve_upload_dataset(req.dataset_path)
    df = load_csv(str(dataset))
    try:
        return run_statistical_test(df, req.spec)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/forecast", dependencies=[Depends(require_api_key)])
def forecast_endpoint(req: ForecastRequest) -> dict[str, Any]:
    """Forecast future values of a numeric column from an uploaded series."""
    from app.tools.forecast_tool import forecast_csv

    dataset = _resolve_upload_dataset(req.dataset_path)
    try:
        return forecast_csv(
            str(dataset),
            column=req.column,
            horizon=req.horizon,
            period=req.period,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _resolve_upload_dataset(dataset_path: str) -> Path:
    """Contain uploaded-dataset references inside uploads/.

    Absolute paths must resolve inside ``uploads/``; relative or arbitrary
    inputs are flattened to a file name inside ``uploads/``. This blocks
    reading arbitrary files from disk (e.g. ``.env``) via API routes.
    """
    uploads_root = settings.upload_dir_path.resolve()
    raw = Path(dataset_path)
    if raw.is_absolute() and not raw.resolve().is_relative_to(uploads_root):
        raise HTTPException(status_code=400, detail="dataset_path must reference an uploaded dataset.")
    dataset = uploads_root / raw.name
    if not dataset.is_file():
        raise HTTPException(status_code=404, detail=f"Uploaded file not found: {raw.name}")
    return dataset


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #
def _execute(run_id: str, req: RunRequest) -> None:
    """Background task that runs the full agent pipeline."""
    store = get_store()
    store.set_status(run_id, "running")

    work_dir_path = settings.upload_dir_path / f"run_{run_id}"
    try:
        initial = {
            "run_id": run_id,
            "dataset_path": str(settings.upload_dir_path / req.filename),
            "work_dir": str(work_dir_path),
            "target": req.target,
            "tune": req.tune,
            "status": "planner_started",
        }
        result = run_pipeline(initial)
        store.set_result(run_id, result)
        _record_artifacts(run_id, result)
    except Exception as exc:  # noqa: BLE001 - surfaced via the API; must catch all pipeline errors
        store.set_status(run_id, "failed", str(exc))
        tracking.log_run(run_id, params={"status": "failed"}, tags={"error": str(exc)})


def _record_artifacts(run_id: str, result: dict[str, Any]) -> None:
    best = result.get("best_model") or {}
    metrics = dict(best.get("test_metrics") or {})
    tracking.log_run(
        run_id,
        params={"target": result.get("target"), "task": result.get("task"), "model": best.get("model")},
        metrics={"cv_mean": best.get("cv_mean"), **{f"test_{k}": v for k, v in metrics.items()}},
        tags={"status": "succeeded"},
    )
    summary = _memory_summary(result)
    if summary:
        memory.embed_store(summary, metadata={"run_id": run_id}, key=f"run:{run_id}")


def _memory_summary(result: dict[str, Any]) -> str:
    best = result.get("best_model") or {}
    eval_summary = result.get("evaluation_summary") or {}
    return "\n".join(
        [
            f"Dataset task: {result.get('task')}",
            f"Best model: {best.get('model')} (CV {best.get('cv_mean')})",
            f"Features: {eval_summary.get('feature_importance')}",
        ]
    )


def _latest_run_id(store) -> str | None:
    runs = store.list_runs(1)
    return runs[0]["run_id"] if runs else None


def _provider_label() -> str:
    from app.services.llm import LLMService

    return LLMService.instance().provider