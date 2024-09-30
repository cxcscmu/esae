from source import workspace
from pathlib import Path

workspace = Path(workspace, "model")
workspace.mkdir(mode=0o770, parents=True, exist_ok=True)
