"""SQL Agent - answers questions about a dataset by writing read-only SQL.

The agent lets the LLM turn a natural-language question into a SELECT statement,
executes it against the dataset with :func:`app.tools.sql_tool.query_dataset`,
then explains the rows back. ``sqlite3`` exceptions and generation failures are
handled so the pipeline keeps running.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from typing import Any

from app.agents.base import BaseAgent
from app.tools.csv_tool import load_csv
from app.tools.sql_tool import query_dataset

logger = logging.getLogger(__name__)

_DEFAULT_QUERY = "SELECT * FROM data LIMIT 10"


class SQLAgent(BaseAgent):
    name = "sql"

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        dataset_path = state.get("dataset_path") or state.get("feature_path")
        if not dataset_path:
            return {"sql_answer": "No dataset available to query.", "status": "sql_done"}

        if not state.get("chat_messages"):
            return {"sql_answer": "No question provided.", "status": "sql_done"}

        question = state["chat_messages"][-1]["content"]
        result = self.answer(question, dataset_path)
        return {
            "sql_query": result.get("query"),
            "sql_answer": result.get("answer"),
            "status": "sql_done",
        }

    def answer(self, question: str, dataset_path: str) -> dict[str, Any]:
        """Return ``{"query": ..., "answer": ...}`` for a natural-language question."""
        query = self._build_query(question, dataset_path)
        try:
            result = query_dataset(dataset_path, query)
        except (ValueError, sqlite3.Error) as exc:
            return {"query": query, "answer": f"Query failed: {exc}"}

        if not result["rows"]:
            return {"query": query, "answer": "The query returned no rows."}

        answer = self.llm.complete(
            "You are the SQL agent for a data science team. Explain the query "
            "result in 2-3 plain sentences.\n\n"
            f"Query: {query}\nColumns: {result['columns']}\nRows: {result['rows']}"
        )
        return {"query": query, "answer": answer}

    def _build_query(self, question: str, dataset_path: str) -> str:
        hint = self._context_hint(dataset_path)
        prompt = (
            "You are an expert SQL writer. Given a dataset schema and a user "
            "question, reply with ONLY a single read-only SELECT statement and "
            "nothing else.\n\n"
            f"Schema: {hint}\nQuestion: {question}\nSQL:"
        )
        candidate = self.llm.complete(prompt).strip()
        return candidate if candidate.lower().startswith("select") else _DEFAULT_QUERY

    @staticmethod
    def _context_hint(dataset_path: str) -> str:
        from app.tools.csv_tool import profile

        prof = profile(load_csv(dataset_path))
        return json.dumps({"columns": prof["column_types"], "rows": prof["rows"]})


__all__ = ["SQLAgent"]