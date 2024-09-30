from pathlib import Path
from source.interpret import workspace

workspace = Path(workspace, "retrieval")
workspace.mkdir(mode=0o770, parents=True, exist_ok=True)
