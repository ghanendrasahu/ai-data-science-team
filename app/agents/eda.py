"""EDA Agent - statistical summaries, correlations and automatic plots."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from app.agents.base import BaseAgent
from app.tools.csv_tool import load_csv, profile
from app.tools.pandas_tool import is_numeric
from app.tools.plot_tool import generate_eda_plots


class EDAAgent(BaseAgent):
    name = "eda"

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        df = load_csv(state["dataset_path"])
        target = state.get("target")

        evals_dir = f"{state['work_dir']}/eda"
        plots = generate_eda_plots(df, target, evals_dir)

        data_profile = state.get("profile") or profile(df)
        summary = self._summarise(df, data_profile, plots)
        narrative = self.llm.complete(self._narrative_prompt(summary))

        return {
            "plots": plots,
            "eda_summary": summary,
            "plan": state.get("plan", "") + "\n\n--- EDA observations ---\n" + narrative,
            "status": "eda_done",
        }

    @staticmethod
    def _summarise(df: pd.DataFrame, data_profile: dict[str, Any], plots) -> str:
        numeric_cols = [c for c in df.columns if is_numeric(df[c])]
        summary = {
            "rows": data_profile["rows"],
            "columns": data_profile["columns"],
            "column_types": data_profile["column_types"],
            "missing_total": int(sum(data_profile["missing"].values())),
            "duplicates": data_profile["duplicates"],
            "numeric_columns": numeric_cols,
            "plots_generated": len(plots),
        }
        return json.dumps(summary, indent=2)

    @staticmethod
    def _narrative_prompt(summary: str) -> str:
        return (
            "You are the EDA agent. Write a concise narrative paragraph (3-5 "
            "sentences) describing the dataset quality, notable distributions and "
            "any red flags based on:\n\n"
            f"{summary}\n\nNarrative:"
        )