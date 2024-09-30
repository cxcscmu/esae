"""
@file: __init__.py
@brief: Initialize the dataset module.
@author: Hao Kang <haok@andrew.cmu.edu>
@date: September 29, 2024
"""

from pathlib import Path

from source import home

home = Path(home, "dataset")
home.mkdir(mode=0o770, exist_ok=True)
