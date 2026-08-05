"""LangGraph orchestration of the AI data science team.

Nodes run sequentially: planner -> eda -> cleaning -> feature -> trainer ->
evaluator -> report. The chatbot node is wired in but is invoked on demand by
the API instead of blocking the pipeline.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from langgraph.graph import END, START, StateGraph

from app.agents.base import BaseAgent
from app.agents.chatbot import ChatbotAgent
from app.agents.cleaning import CleaningAgent
from app.agents.eda import EDAAgent
from app.agents.evaluator import EvaluatorAgent
from app.agents.feature import FeatureAgent
from app.agents.planner import PlannerAgent
from app.agents.report import ReportAgent
from app.agents.trainer import TrainerAgent
from app.graph.state import WorkflowState

_AGENTS: dict[str, BaseAgent] = {
    "planner": PlannerAgent(),
    "eda": EDAAgent(),
    "cleaning": CleaningAgent(),
    "feature": FeatureAgent(),
    "trainer": TrainerAgent(),
    "evaluator": EvaluatorAgent(),
    "report": ReportAgent(),
    "chatbot": ChatbotAgent(),
}


@lru_cache(maxsize=1)
def build_graph():
    """Compile the state machine once and reuse it across requests."""
    builder = StateGraph(WorkflowState)

    for name, agent in _AGENTS.items():
        builder.add_node(name, agent)

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "eda")
    builder.add_edge("eda", "cleaning")
    builder.add_edge("cleaning", "feature")
    builder.add_edge("feature", "trainer")
    builder.add_edge("trainer", "evaluator")
    builder.add_edge("evaluator", "report")
    builder.add_edge("report", END)

    return builder.compile()


def run_pipeline(initial_state: dict[str, Any]) -> dict[str, Any]:
    """Execute the full pipeline for one dataset and return the final state."""
    result = build_graph().invoke(initial_state)
    return dict(result)


def agent_names() -> list[str]:
    return list(_AGENTS.keys())