from pathlib import Path

workspace = Path("/data/group_data/cx_group/esae")
workspace.mkdir(mode=0o770, parents=True, exist_ok=True)

import os

os.environ["HF_HOME"] = Path(workspace, "huggingface").as_posix()

from rich.console import Console

console = Console(width=80)
console._log_render.show_path = False
console._log_render.omit_repeated_times = False

import warnings

warnings.filterwarnings("ignore")
