from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def execute_cells(path: Path, indices: list[int]) -> dict[str, Any]:
    """Execute selected official notebook cells in one isolated namespace."""

    notebook = json.loads(path.read_text(encoding="utf-8"))
    namespace: dict[str, Any] = {
        "__name__": "vendored_sosmc_notebook",
        "__file__": str(path),
    }
    for index in indices:
        cell = notebook["cells"][index]
        if cell["cell_type"] != "code":
            raise ValueError(f"Cell {index} is not code")
        source = "".join(cell["source"])
        exec(compile(source, f"{path}:cell-{index}", "exec"), namespace)
    return namespace

