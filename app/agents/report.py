"""Report Agent - generates technical and business insights in Markdown."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.agents.base import BaseAgent
from app.config import get_settings


class ReportAgent(BaseAgent):
    name = "report"

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        context = self._build_context(state)
        technical = self.llm.complete(
            "You are the technical report agent. Write a detailed technical report "
            "covering the dataset, preprocessing, models compared and evaluation "
            "metrics.\n\n"
            f"Context:\n{context}"
        )
        business = self.llm.complete(
            "You are the business analyst. Write a short business summary: key "
            "findings, value delivered and next steps for non-technical readers.\n\n"
            f"Context:\n{context}"
        )

        report = self._render_report(state, technical, business)
        report_path = self._write_report(state["run_id"], report)

        return {
            "report_path": str(report_path),
            "report_markdown": report,
            "status": "report_done",
        }

    @staticmethod
    def _build_context(state: dict[str, Any]) -> str:
        return json.dumps(
            {
                "dataset": state.get("dataset_path"),
                "target": state.get("target"),
                "profile": state.get("profile"),
                "cleaning": state.get("cleaning_summary"),
                "features": state.get("feature_summary"),
                "best_model": state.get("best_model"),
                "evaluation": state.get("evaluation_summary"),
            },
            indent=2,
            default=str,
        )

    @staticmethod
    def _render_report(state: dict[str, Any], technical: str, business: str) -> str:
        best = state.get("best_model", {})
        lines = [
            "# AI Data Science Team - Analysis Report",
            "",
            f"- **Run ID**: `{state.get('run_id')}`",
            f"- **Dataset**: `{state.get('dataset_path')}`",
            f"- **Target**: `{state.get('target')}`",
            f"- **Best model**: `{best.get('model')}` (CV score {best.get('cv_mean')})",
            "",
            "## Technical Report",
            "",
            technical.strip(),
            "",
            "## Business Insights",
            "",
            business.strip(),
            "",
            "---",
            "",
            f"*Generated automatically by the AI Data Science Team.*",
            "",
        ]
        return "\n".join(lines)

    @staticmethod
    def _write_report(run_id: str, report: str) -> Path:
        out_dir = get_settings().report_dir_path
        path = out_dir / f"report_{run_id}.md"
        path.write_text(report, encoding="utf-8")
        return path