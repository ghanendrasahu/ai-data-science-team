"""Automatic visualisation generation for EDA.

Plots are written as PNG files so the frontend and reports can render them
without holding a live figure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from app.tools.pandas_tool import is_numeric


def _safe_title(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in text)


def generate_eda_plots(
    df: pd.DataFrame, target: str | None, out_dir: str | Path, top_n: int = 6
) -> list[dict[str, Any]]:
    """Render histograms, boxplots and target analysis; return plot metadata."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plots: list[dict[str, Any]] = []

    numeric_cols = [c for c in df.columns if is_numeric(df[c])]
    categorical_cols = [c for c in df.columns if not is_numeric(df[c])]

    if numeric_cols:
        plots.extend(
            _distribution_plots(df, numeric_cols[:top_n], out_dir, "dist")
        )
        plots.extend(
            _correlation_plot(df, numeric_cols, out_dir)
        )

    if categorical_cols:
        plots.extend(
            _category_plots(df, categorical_cols[:top_n], out_dir, "cat")
        )

    if target and target in df.columns and numeric_cols:
        plots.extend(
            _target_analysis(df, target, numeric_cols[:top_n], out_dir)
        )

    return plots


def _distribution_plots(df, cols, out_dir, prefix) -> list[dict[str, Any]]:
    plots = []
    for col in cols:
        fig, ax = plt.subplots(figsize=(7, 4))
        df[col].dropna().hist(bins=30, ax=ax, color="#4C78A8", edgecolor="white")
        ax.set_title(f"Distribution - {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")
        path = out_dir / f"{prefix}_{_safe_title(col)}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=100)
        plt.close(fig)
        plots.append({"type": "histogram", "column": col, "path": str(path)})
    return plots


def _correlation_plot(df, cols, out_dir) -> list[dict[str, Any]]:
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)))
    ax.set_yticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(corr.columns, fontsize=8)
    ax.set_title("Correlation Matrix")
    fig.colorbar(im, ax=ax, shrink=0.8)
    path = out_dir / "correlation_matrix.png"
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return [{"type": "correlation", "column": "all", "path": str(path)}]


def _category_plots(df, cols, out_dir, prefix) -> list[dict[str, Any]]:
    plots = []
    for col in cols:
        counts = df[col].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(7, 4))
        counts.plot(kind="bar", ax=ax, color="#54A24B")
        ax.set_title(f"Category counts - {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        path = out_dir / f"{prefix}_{_safe_title(col)}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=100)
        plt.close(fig)
        plots.append({"type": "bar", "column": col, "path": str(path)})
    return plots


def _target_analysis(df, target, cols, out_dir) -> list[dict[str, Any]]:
    plots = []
    for col in cols:
        if col == target:
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        df[[col, target]].dropna().boxplot(column=col, by=target, ax=ax, grid=False)
        ax.set_title(f"{col} by {target}")
        ax.set_xlabel(target)
        ax.set_ylabel(col)
        path = out_dir / f"target_{_safe_title(col)}.png"
        fig.suptitle("")
        fig.tight_layout()
        fig.savefig(path, dpi=100)
        plt.close(fig)
        plots.append({"type": "boxplot", "column": col, "target": target, "path": str(path)})
    return plots