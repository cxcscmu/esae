"""
@file: WebQa.py
@brief: Implementation of the WebQa dataset.
@author: Hao Kang <haok@andrew.cmu.edu>
@date: September 29, 2024
"""

import gdown
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


class WebQaDownload:
    """
    Download the WebQa dataset.
    """

    def __init__(self) -> None:
        self.home = Path(home, self.__class__.__name__)
        self.home.mkdir(mode=0o770, exist_ok=True)

    def step1(self) -> None:
        """
        Download and extract the images.

        Automatic download with gdrive may fail due to connection issues.
        Alternatively, manually download the images from the following link:
        https://drive.google.com/drive/folders/19ApkbD5w0I5sV1IeQ9EofJRyAjKnA7tb
        """
        gdown.download_folder(
            id="19ApkbD5w0I5sV1IeQ9EofJRyAjKnA7tb",
            output=self.home.as_posix(),
            remaining_ok=True,
            resume=True,
        )
        # there is a hard limit on 50 files per folder
        # we manually add the last file to download
        gdown.download(
            id="1SlYNpYYpwTfxIjQIDlM3o7a8kcpKO8PB",
            output=Path(self.home, "imgs.7z.051").as_posix(),
            resume=True,
        )
        # the final `imgs.tsv` contains (id, base64) pairs
        # inefficient, but compatible with the MARVEL codebase
        with console.status("Extracting..."):
            subprocess.run(
                ["7z", "x", "imgs.7z.001"],
                cwd=self.home,
                capture_output=True,
                check=True,
            )
        # remove the 7z files to save space
        for file in self.home.glob("*.7z.*"):
            file.unlink()

    def dispatch(self) -> None:
        """
        Dispatch the download steps.
        """
        # self.step1()


def main():
    """
    Initialize the WebQa dataset.
    """
    webQaDownload = WebQaDownload()
    webQaDownload.dispatch()


if __name__ == "__main__":
    main()
