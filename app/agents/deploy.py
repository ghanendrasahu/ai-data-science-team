"""Deployment Agent - packages the winning model for production serving.

Copies the trained pipeline, target metadata and a minimal predictor into a
self-contained directory under the configured deploy directory so the model can
be served by the ``/predict`` endpoint or imported elsewhere.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from app.agents.base import BaseAgent
from app.config import get_settings


class DeploymentAgent(BaseAgent):
    name = "deploy"

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        best = state.get("best_model")
        if not best or not best.get("path"):
            return {"deployment_summary": {"error": "No trained model to deploy."}}

        target = state.get("target")
        task = state.get("task") or best.get("task")
        run_id = state["run_id"]

        bundle = _build_bundle(run_id, best["path"], task, target, state)
        return {
            "deployment_path": bundle["path"],
            "deployment_summary": bundle,
            "status": "deploy_done",
        }


def _build_bundle(
    run_id: str,
    model_path: str,
    task: str | None,
    target: str | None,
    state: dict[str, Any],
) -> dict[str, Any]:
    settings = get_settings()
    out_dir = settings.deploy_dir_path / run_id
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    src = Path(model_path)
    artifact = out_dir / "model.joblib"
    shutil.copy2(src, artifact)

    schema = {
        "run_id": run_id,
        "task": task,
        "target": target,
        "feature_names": _feature_names_from_state(state),
        "model_path": str(artifact),
        "created_at": __import__("datetime").datetime.now().isoformat(),
    }
    (out_dir / "schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
    (out_dir / "predict.py").write_text(_PREDICTOR_SOURCE, encoding="utf-8")
    (out_dir / "README.md").write_text(_bundle_readme(schema), encoding="utf-8")

    return {
        "path": str(out_dir),
        "schema": schema,
        "artifacts": [str(artifact), "schema.json", "predict.py", "README.md"],
    }


def _feature_names_from_state(state: dict[str, Any]) -> list[str]:
    training = state.get("training_summary") or {}
    names = training.get("feature_names") or state.get("feature_names") or []
    return list(names)


_PREDICTOR_SOURCE = '''"""Self-contained predictor for a deployed AI Data Science Team model."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

HERE = Path(__file__).parent
MODEL = joblib.load(HERE / "model.joblib")


def _load_schema() -> dict:
    return json.loads((HERE / "schema.json").read_text(encoding="utf-8"))


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the training-time feature encoding (numeric keep, categorical one-hot)."""
    schema = _load_schema()
    wanted = schema.get("feature_names") or []

    target = schema.get("target")
    if target and target in df.columns:
        df = df.drop(columns=[target])

    X = df.copy()
    numeric = [col for col in X.columns if pd.api.types.is_numeric_dtype(X[col])]
    parts = [X[numeric].reset_index(drop=True)]
    for col in X.columns:
        if col in numeric:
            continue
        if X[col].nunique() <= 50:
            parts.append(pd.get_dummies(X[col], prefix=col, drop_first=False))
    prepared = pd.concat(parts, axis=1)

    if wanted:
        missing = [col for col in wanted if col not in prepared.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")
        prepared = prepared[wanted]
    return prepared


def predict_csv(csv_path: str | Path) -> list:
    df = pd.read_csv(csv_path)
    X = _prepare(df)
    return [float(v) if hasattr(v, "item") else v for v in MODEL["pipeline"].predict(X)]


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("usage: python predict.py <dataset.csv>", file=sys.stderr)
        raise SystemExit(1)
    print(json.dumps(predict_csv(sys.argv[1])))
'''


def _bundle_readme(schema: dict[str, Any]) -> str:
    return (
        "# Deployed Model\n\n"
        f"- Run: `{schema['run_id']}`\n"
        f"- Task: `{schema['task']}`\n"
        f"- Target: `{schema['target']}`\n"
        "- Artifacts: `model.joblib`, `schema.json`, `predict.py`\n\n"
        "Run predictions on a CSV:\n\n"
        "```bash\npython predict.py new_data.csv\n```\n"
    )


__all__ = ["DeploymentAgent", "_build_bundle"]