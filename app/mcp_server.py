"""MCP server that exposes the AI Data Science Team tools.

Runs standalone over stdio (works with any MCP client / Claude Desktop):

    .venv\\Scripts\\python.exe -m app.mcp_server

Requires the optional ``mcp`` package (mcp>=2). The rest of the app never
imports this module at boot, so the feature is isolated and optional.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

try:
    from mcp.server import mcpserver  # type: ignore[import-not-found]

    _MCP_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    mcpserver = None  # type: ignore[assignment, misc]
    _MCP_AVAILABLE = False


def build_server(name: str = "ai-data-science-team") -> Any:
    """Build and register the MCP tools; returns the underlying server."""
    if not _MCP_AVAILABLE or mcpserver is None:
        raise RuntimeError("MCP support requires the optional `mcp` package (v2).")

    server = mcpserver.MCPServer(name=name, version="0.1.0")

    @server.tool()
    def profile_dataset(csv_path: str) -> str:
        """Return a JSON profile (rows, columns, missing, cardinality) of a CSV."""
        from app.tools.csv_tool import load_csv, profile

        return json.dumps(profile(load_csv(csv_path)), default=str)

    @server.tool()
    def run_sql(csv_path: str, sql: str) -> str:
        """Execute a read-only SELECT against a CSV and return the rows."""
        from app.tools.sql_tool import query_dataset

        try:
            result = query_dataset(csv_path, sql)
        except Exception as exc:  # noqa: BLE001 - surfaced to the MCP client
            return f"error: {exc}"
        return json.dumps(result, default=str)

    @server.tool()
    def get_run(run_id: str) -> str:
        """Return a finished workflow run's result JSON."""
        from app.services.store import get_store

        record = get_store().get_run(run_id)
        if record is None:
            return "error: run not found"
        return json.dumps(record, default=str)

    @server.tool()
    def list_recent_runs(limit: int = 5) -> str:
        """List the most recent workflow runs."""
        from app.services.store import get_store

        return json.dumps(get_store().list_runs(limit), default=str)

    @server.tool()
    def feature_importance(best_model_path: str) -> str:
        """Return feature importances for a saved best model."""
        from app.tools.sklearn_tool import feature_importance

        try:
            return json.dumps(feature_importance(best_model_path), default=str)
        except Exception as exc:  # noqa: BLE001
            return f"error: {exc}"

    return server


def main() -> None:
    _run_stdio(build_server())

def _run_stdio(server: Any) -> None:
    import anyio
    from mcp.server.stdio import stdio_server  # type: ignore[import-not-found]

    async def _serve() -> None:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream)

    anyio.run(_serve)


if __name__ == "__main__":
    main()