from pathlib import Path
from source import workspace

workspace = Path(workspace, "dataset")
workspace.mkdir(mode=0o770, parents=True, exist_ok=True)

from source.dataset.msMarco import MsMarcoDataset
from source.dataset.beir import BeirDataset
