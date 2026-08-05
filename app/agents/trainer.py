"""ML Trainer Agent - trains and compares a zoo of models."""

from __future__ import annotations

import json
from typing import Any

from app.agents.base import BaseAgent
from app.tools.csv_tool import load_csv
from app.tools.pandas_tool import prepare_for_ml
from app.tools.sklearn_tool import detect_task, train_all


class TrainerAgent(BaseAgent):
    name = "trainer"

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        source = state.get("feature_path") or state["dataset_path"]
        df = load_csv(source)
        target = state["target"]

        if target not in df.columns:
            raise ValueError(f"Target column '{target}' missing after feature engineering.")

        X, y, feature_names = prepare_for_ml(df, target)
        task = detect_task(y)

        training = train_all(
            X=X,
            y=y,
            task=task,
            model_dir=f"{state['work_dir']}/models",
            tune=state.get("tune", False),
        )

        narrative = self.llm.complete(
            "You are the ML trainer agent. Summarise the training run in one "
            "sentence.\n\n"
            f"Results: {json.dumps(training, indent=2, default=str)}"
        )

        return {
            "training_summary": {**training, "narrative": narrative},
            "models_trained": training["results"],
            "best_model": training["best_model"],
            "task": task,
            "status": "trainer_done",
        }