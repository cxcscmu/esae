"""
@file: __init__.py
@brief: Initialize the project.
@author: Hao Kang <haok@andrew.cmu.edu>
@date: September 29, 2024
"""

from pathlib import Path

home = Path("/data/group_data/cx_group/esae")
home.mkdir(mode=0o770, parents=True, exist_ok=True)

from rich.console import Console

console = Console(width=120)
console._log_render.show_path = False
console._log_render.omit_repeated_times = False
