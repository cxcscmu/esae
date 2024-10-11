from pathlib import Path
from source import workspace

workspace = Path(workspace, "dataset")
workspace.mkdir(mode=0o770, parents=True, exist_ok=True)
