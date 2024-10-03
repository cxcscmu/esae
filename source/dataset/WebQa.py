"""
@file: WebQa.py
@brief: Implementation of the WebQa dataset.
@author: Hao Kang <haok@andrew.cmu.edu>
@date: September 29, 2024
"""

import gdown
import requests
import subprocess
from pathlib import Path

from source import console
from source.dataset import home
from source.interface import Txt2ImgDataset


class WebQa(Txt2ImgDataset):
    """
    Implementation of the WebQa dataset.
    """

    def __init__(self) -> None:
        self.home = Path(home, self.__class__.__name__)
        self.home.mkdir(mode=0o770, exist_ok=True)
