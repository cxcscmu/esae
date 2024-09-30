"""
@file: dataset.py
@brief: Abstract classes for datasets.
@author: Hao Kang <haok@andrew.cmu.edu>
@date: September 29, 2024
"""

from pathlib import Path
from abc import ABC, abstractmethod


class Txt2TxtDataset(ABC):
    """
    The interface for a text-to-text dataset.
    Queries and documents are both represented as text.

    Attributes:
        home: The root directory of the dataset.
    """

    home: Path

    def __init__(self) -> None:
        raise NotImplementedError


class Txt2ImgDataset(ABC):
    """
    The interface for a text-to-image dataset.
    Queries are represented as text, and documents are represented as images.

    Attributes:
        home: The root directory of the dataset.
    """

    home: Path

    def __init__(self) -> None:
        raise NotImplementedError
