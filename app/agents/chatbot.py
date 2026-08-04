"""Dataset Chat Agent - answers questions about the analysed dataset."""

from __future__ import annotations

import json
from typing import Any

from app.agents.base import BaseAgent


class ChatbotAgent(BaseAgent):
    name = "chatbot"

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        """Answer the last user question using the pipeline context.

        The workflow graph keeps this node reachable but it is primarily used
        directly by the FastAPI chat endpoint via :meth:`answer`.
        """
        if not state.get("chat_messages"):
            return {"chat_answer": "No question provided.", "status": "chat_done"}

        question = state["chat_messages"][-1]["content"]
        answer = self.answer(question, state)
        return {
            "chat_answer": answer,
            "chat_messages": [{"role": "assistant", "content": answer}],
            "status": "chat_done",
        }

    def answer(self, question: str, context: dict[str, Any]) -> str:
        prompt = (
            "You are the dataset chat assistant for the AI Data Science Team. "
            "Answer the user's question using ONLY the analysis context below. "
            "Be concise and clear.\n\n"
            f"Context:\n{self._context_string(context)}\n\n"
            f"Question: {question}"
        )
        return self.llm.complete(prompt)

    @staticmethod
    def _context_string(state: dict[str, Any]) -> str:
        return json.dumps(
            {
                "profile": state.get("profile"),
                "plan": state.get("plan"),
                "eda": state.get("eda_summary"),
                "cleaning": state.get("cleaning_summary"),
                "features": state.get("feature_summary"),
                "best_model": state.get("best_model"),
                "evaluation": state.get("evaluation_summary"),
                "report": state.get("report_markdown", "")[:2000],
            },
            indent=2,
            default=str,
        )