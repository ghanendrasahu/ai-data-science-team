"""Feature Engineering Agent - encoding, scaling and feature importance."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from app.agents.base import BaseAgent
from app.tools.csv_tool import load_csv, save_csv
from app.tools.pandas_tool import is_numeric, remove_constant_columns


class FeatureAgent(BaseAgent):
    name = "feature"

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        source = state.get("cleaned_path") or state["dataset_path"]
        df = load_csv(source)
        target = state.get("target")

        df = remove_constant_columns(df)
        summary = self._encode_categoricals(df, target)
        self._scale_numerics(df, target)

        if target and target in df.columns:
            summary["target"] = target

        feature_path = f"{state['work_dir']}/featured.csv"
        save_csv(df, feature_path)

        narrative = self.llm.complete(
            "You are the feature engineering agent. Summarise the encoding and "
            "scaling choices made and suggest up to 3 new features.\n\n"
            f"Summary: {json.dumps(summary, indent=2)}"
        )

        return {
            "feature_path": feature_path,
            "feature_summary": {**summary, "narrative": narrative},
            "status": "feature_done",
        }

    @staticmethod
    def _encode_categoricals(df: pd.DataFrame, target: str | None) -> dict[str, Any]:
        encoded = []
        for col in df.columns:
            if is_numeric(df[col]) or col == target:
                continue
            if df[col].nunique() <= 50:
                df[col] = pd.Series(np.asarray(LabelEncoder().fit_transform(df[col])), index=df.index)
                encoded.append(col)
            else:
                df[col] = df[col].astype("category").cat.codes
        return {"categorical_columns_encoded": encoded, "numeric_columns": len(df.columns)}

    @staticmethod
    def _scale_numerics(df: pd.DataFrame, target: str | None) -> None:
        numeric_cols = [c for c in df.columns if is_numeric(df[c]) and c != target]
        if not numeric_cols:
            return
        df[numeric_cols] = StandardScaler().fit_transform(df[numeric_cols])