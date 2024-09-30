import aiofiles
from pathlib import Path
from aiohttp import ClientSession, ClientTimeout
from rich.progress import Progress
from source import console


async def download(url: str, file: Path, timeout: int = 30 * 60) -> None:
    """
    Download the file from the given URL to the specified file path.

    :type url: str
    :param url: The URL to download the file from.
    :type file: Path
    :param file: The file path to save the downloaded file.
    :type timeout: int
    :param timeout: The timeout in seconds for the download.
    """
    async with ClientSession(timeout=ClientTimeout(timeout)) as session:
        async with session.get(url) as response:
            response.raise_for_status()
            with Progress(console=console) as progress:
                T = progress.add_task("Download:", total=response.content_length)
                async with aiofiles.open(file, "wb") as f:
                    async for chunk in response.content.iter_any():
                        await f.write(chunk)
                        progress.advance(T, len(chunk))
                progress.remove_task(T)
