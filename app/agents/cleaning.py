"""Cleaning Agent - missing values, duplicates, outliers and transformations."""

from __future__ import annotations

from typing import Any

import pandas as pd

from app.agents.base import BaseAgent
from app.tools.csv_tool import load_csv, save_csv
from app.tools.pandas_tool import drop_duplicates, fill_missing, winsorize_outliers


class CleaningAgent(BaseAgent):
    name = "cleaning"

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        df = load_csv(state["dataset_path"])
        before_rows = len(df)
        before_missing = int(df.isna().sum().sum())

        df = drop_duplicates(df)
        df = fill_missing(df)
        df = winsorize_outliers(df)

        cleaned_path = f"{state['work_dir']}/cleaned.csv"
        save_csv(df, cleaned_path)

        summary = {
            "rows_before": before_rows,
            "rows_after": int(len(df)),
            "missing_before": before_missing,
            "missing_after": int(df.isna().sum().sum()),
            "outliers_capped": True,
            "saved_to": cleaned_path,
        }

        narrative = self.llm.complete(
            "You are the data cleaning agent. Summarise the cleaning operations "
            "just performed in one sentence.\n\n"
            f"Summary: {summary}"
        )

        return {
            "cleaned_path": cleaned_path,
            "cleaning_summary": {**summary, "narrative": narrative},
            "status": "cleaning_done",
        }