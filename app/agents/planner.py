"""Planner Agent - analyses the dataset profile and decides the workflow plan."""

from __future__ import annotations

import json
from typing import Any

from app.agents.base import BaseAgent
from app.tools.csv_tool import load_csv, profile


class PlannerAgent(BaseAgent):
    name = "planner"

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        df = load_csv(state["dataset_path"])
        data_profile = profile(df)

        target = state.get("target") or data_profile["suggested_target"]

        prompt = self._plan_prompt(data_profile)
        plan = self.llm.complete(prompt)

        update: dict[str, Any] = {
            "profile": data_profile,
            "target": target,
            "plan": plan,
            "status": "planner_done",
        }
        try:
            parsed = self.llm.complete_json(self._target_prompt(data_profile))
            if isinstance(parsed, dict) and parsed.get("target"):
                update["target"] = str(parsed["target"])
        except Exception:
            pass
        return update

    @staticmethod
    def _plan_prompt(profile: dict[str, Any]) -> str:
        return (
            "You are the planner of an autonomous data science team. "
            "Given the dataset profile below, produce a short step-by-step plan "
            "covering: EDA, cleaning, feature engineering, model training, "
            "evaluation and reporting.\n\n"
            f"Profile:\n{json.dumps(profile, indent=2)}\n\nPlan:"
        )

    @staticmethod
    def _target_prompt(profile: dict[str, Any]) -> str:
        return (
            "From this dataset profile, identify the best target column for a "
            "supervised ML task. Reply ONLY with JSON like "
            '{"target": "column_name"}.\n\n'
            f"Profile:\n{json.dumps(profile, indent=2)}"
        )