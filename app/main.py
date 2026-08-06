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
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.agents.chatbot import ChatbotAgent
from app.config import get_settings
from app.graph.workflow import agent_names, run_pipeline
from app.services.store import get_store

settings = get_settings()
app = FastAPI(title=settings.app_name, version=settings.app_version)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_chatbot = ChatbotAgent()


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


# --------------------------------------------------------------------------- #
# Lifecycle
# --------------------------------------------------------------------------- #
@app.on_event("startup")
def _ensure_dirs() -> None:
    settings.upload_dir_path.mkdir(parents=True, exist_ok=True)
    settings.model_dir_path.mkdir(parents=True, exist_ok=True)
    settings.report_dir_path.mkdir(parents=True, exist_ok=True)
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
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/upload")
async def upload(file: UploadFile = File(...)) -> dict[str, Any]:  # noqa: B008 - FastAPI injects the file
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    destination = settings.upload_dir_path / Path(file.filename).name
    with destination.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {"filename": destination.name, "path": str(destination)}


@app.post("/workflow", response_model=RunResponse)
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


@app.post("/chat")
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


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #
def _execute(run_id: str, req: RunRequest) -> None:
    """Background task that runs the full agent pipeline."""
    store = get_store()
    store.set_status(run_id, "running")

    work_dir = settings.upload_dir_path / f"run_{run_id}"
    try:
        initial = {
            "run_id": run_id,
            "dataset_path": str(settings.upload_dir_path / req.filename),
            "work_dir": str(work_dir),
            "target": req.target,
            "tune": req.tune,
            "status": "planner_started",
        }
        result = run_pipeline(initial)
        store.set_result(run_id, result)
    except Exception as exc:  # noqa: BLE001 - surfaced via the API; must catch all pipeline errors
        store.set_status(run_id, "failed", str(exc))


def _provider_label() -> str:
    from app.services.llm import LLMService

    return LLMService.instance().provider