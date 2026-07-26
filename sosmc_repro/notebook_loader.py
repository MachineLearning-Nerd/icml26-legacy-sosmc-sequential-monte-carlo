from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any


def execute_cells(path: Path, indices: list[int]) -> dict[str, Any]:
    """Execute selected official notebook cells in one isolated namespace."""

    notebook = json.loads(path.read_text(encoding="utf-8"))
    module_name = "vendored_sosmc_notebook"
    module = types.ModuleType(module_name)
    module.__file__ = str(path)
    sys.modules[module_name] = module
    namespace = module.__dict__
    for index in indices:
        cell = notebook["cells"][index]
        if cell["cell_type"] != "code":
            raise ValueError(f"Cell {index} is not code")
        source = "".join(cell["source"])
        exec(compile(source, f"{path}:cell-{index}", "exec"), namespace)
    return namespace
