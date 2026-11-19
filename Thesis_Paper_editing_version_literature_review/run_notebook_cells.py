from pathlib import Path

import nbformat
from nbclient import NotebookClient


notebook_path = Path("t.ipynb")
notebook = nbformat.read(notebook_path, as_version=4)

new_cells = [
    """from datetime import datetime\nrun_started = datetime.now()\nprint(f\"Cell 1/5 — kernel started at {run_started:%H:%M:%S}\")""",
    """numbers = [1, 2, 3, 4, 5]\nprint(f\"Cell 2/5 — numbers: {numbers}\")""",
    """squares = [number ** 2 for number in numbers]\nprint(f\"Cell 3/5 — squares: {squares}\")""",
    """square_total = sum(squares)\nprint(f\"Cell 4/5 — sum of squares: {square_total}\")""",
    """run_summary = {\n    \"cell_count\": 5,\n    \"input\": numbers,\n    \"output\": squares,\n    \"total\": square_total,\n}\nprint(f\"Cell 5/5 — completed: {run_summary}\")""",
]

for source in new_cells:
    notebook.cells.append(nbformat.v4.new_code_cell(source=source))

notebook.metadata.kernelspec = {
    "display_name": "Codex Notebook Python",
    "language": "python",
    "name": "codex-notebook",
}

client = NotebookClient(
    notebook,
    kernel_name="codex-notebook",
    timeout=60,
    resources={"metadata": {"path": str(notebook_path.parent.resolve())}},
)
client.execute()
nbformat.write(notebook, notebook_path)

for index, cell in enumerate(notebook.cells[-5:], start=1):
    output_text = "".join(
        output.get("text", "")
        for output in cell.get("outputs", [])
        if output.get("output_type") == "stream"
    ).strip()
    print(f"[{index}/5] {output_text}", flush=True)
