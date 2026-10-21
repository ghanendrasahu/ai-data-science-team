"""Evaluation Agent - compares models, analyses errors and recommends a winner."""

from __future__ import annotations

import json
from typing import Any

from app.agents.base import BaseAgent
from app.tools.csv_tool import load_csv
from app.tools.sklearn_tool import feature_importance, permutation_summary


class EvaluatorAgent(BaseAgent):
    name = "evaluator"

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        training = state.get("training_summary")
        if not training:
            raise ValueError("No training summary found; run the trainer first.")

        best = training["best_model"]
        results = training["results"]

        importances = {}
        model_path = best.get("path")
        if model_path:
            try:
                importances = feature_importance(model_path)
            except (OSError, ValueError, KeyError, EOFError):
                importances = {}

        # Model-agnostic explainability - covers every estimator in the zoo.
        permutation = {}
        if model_path:
            source = state.get("feature_path") or state.get("dataset_path")
            target = state.get("target")
            if source and target:
                try:
                    df = load_csv(source)
                    if target in df.columns:
                        permutation = permutation_summary(model_path, df, df[target])
                except (OSError, ValueError, KeyError, EOFError, RuntimeError):
                    permutation = {}

        evaluation = {
            "best_model": best.get("model"),
            "best_cv_score": best.get("cv_mean"),
            "test_metrics": best.get("test_metrics", {}),
            "model_count": len([r for r in results if "error" not in r]),
            "leaderboard": sorted(
                (r for r in results if "error" not in r),
                key=lambda r: r["cv_mean"],
                reverse=True,
            ),
            "feature_importance": importances,
            "permutation_importance": permutation,
        }

        narrative = self.llm.complete(
            "You are the model evaluation agent. Explain why the winning model was "
            "chosen and any caveats.\n\n"
            f"Evaluation: {json.dumps(evaluation, indent=2, default=str)}"
        )

        return {
            "evaluation_summary": {**evaluation, "narrative": narrative},
            "best_model": best,
            "status": "evaluator_done",
        }