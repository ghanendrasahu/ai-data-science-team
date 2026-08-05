"""LangGraph state definition shared by all workflow nodes."""

from __future__ import annotations

from typing import Annotated, Any, TypedDict


def append_reducer(left: list[Any], right: list[Any]) -> list[Any]:
    """Standard append reducer for list-typed state keys."""
    if left is None:
        left = []
    if right is None:
        return left
    return [*left, *right]


def replace_reducer(left: Any, right: Any) -> Any:
    """Default reducer that overwrites the previous value."""
    return right


class WorkflowState(TypedDict, total=False):
    """State flowing through the agent graph.

    Each node reads the inputs it needs and writes back only the keys it owns.
    DataFrames are intentionally not stored in state: agents reload the latest
    CSV from disk, which keeps state small and serialisable.
    """

    run_id: str
    dataset_path: str
    work_dir: str
    target: str
    task: str
    tune: bool

    profile: dict[str, Any]
    plan: str
    status: str
    error: str

    eda_summary: str
    plots: Annotated[list[dict[str, Any]], append_reducer]

    cleaned_path: str
    cleaning_summary: dict[str, Any]

    feature_path: str
    feature_summary: dict[str, Any]

    training_summary: dict[str, Any]
    models_trained: Annotated[list[dict[str, Any]], append_reducer]

    evaluation_summary: dict[str, Any]
    best_model: dict[str, Any]

    report_path: str
    report_markdown: str

    chat_messages: Annotated[list[dict[str, str]], append_reducer]
    chat_answer: str


__all__ = ["WorkflowState", "append_reducer", "replace_reducer"]