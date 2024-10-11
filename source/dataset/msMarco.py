import faiss
import torch
import asyncio
import numpy as np
import subprocess
import pyarrow as pa
import pyarrow.parquet as pq
from collections import deque
import torch.cuda as cuda
from pathlib import Path
from typing import Iterator, List, Type, Tuple
from torch import Tensor
from torch import FloatStorage, IntStorage
from torch import FloatTensor, IntTensor
from torch.utils.data import TensorDataset, ConcatDataset, DataLoader
from rich.progress import Progress
from source import console
from source.interface import Dataset, Embedding, PartitionName
from source.dataset import workspace
from source.utilities.dataset import download
from source.embedding.bgeBase import BgeBaseEmbedding
from source.embedding.miniPcm import MiniPcmEmbedding


class MsMarcoDataset(Dataset):
    """
    Implementation of Dataset interface for MsMarco.

    MsMarco/
    ├── doc
    │   ├── partition-00000000.parquet
    │   ├── partition-00000001.parquet
    │   ├── partition-00000002.parquet
    │   └── partition-00000003.parquet
    ├── docEmb
    │   └── BgeBase
    │       ├── partition-00000000.bin
    │       ├── partition-00000001.bin
    │       ├── partition-00000002.bin
    │       └── partition-00000003.bin
    ├── mixEmb
    │   └── BgeBase
    │       ├── Train.bin
    │       └── Validate.bin
    ├── qry
    │   ├── Train.parquet
    │   └── Validate.parquet
    ├── qryEmb
    │   └── BgeBase
    │       ├── Train.bin
    │       └── Validate.bin
    └── qryRel
        ├── Train.tsv
        └── Validate.tsv
    """

    name = "MsMarco"

    def didIter(self, batchSize: int) -> Iterator[List[int]]:
        for i in range(DocIterInit.N):
            file = pq.ParquetFile(Path(DocIterInit.base, f"partition-{i:08d}.parquet"))
            for part in file.iter_batches(batchSize, columns=["id"]):
                yield part.column("id").to_pylist()

    def docIter(self, batchSize: int) -> Iterator[List[str]]:
        for i in range(DocIterInit.N):
            file = pq.ParquetFile(Path(DocIterInit.base, f"partition-{i:08d}.parquet"))
            for part in file.iter_batches(batchSize, columns=["text"]):
                yield part.column("text").to_pylist()

    def docEmbIter(
        self,
        embedding: Type[Embedding],
        batchSize: int,
        numWorkers: int,
        shuffle: bool,
    ) -> Iterator[Tensor]:
        partitions: List[TensorDataset] = []
        for i in range(DocIterInit.N):
            path = Path(DocIterInit.base, f"partition-{i:08d}.parquet")
            file = pq.ParquetFile(path)
            N, D = file.metadata.num_rows, embedding.size
            path = Path(DocEmbIterInit.base, embedding.name, f"partition-{i:08d}.bin")
            storage = FloatStorage.from_file(str(path), False, N * D)
            samples = FloatTensor(storage).reshape(N, D)
            partitions.append(TensorDataset(samples))
        dataloader: DataLoader = DataLoader(
            ConcatDataset(partitions),
            batchSize,
            shuffle=shuffle,
            num_workers=numWorkers,
        )
        for batch in dataloader:
            yield batch[0]
        del partitions

    def docPrefixEmbIter(
        self,
        embedding: Type[Embedding],
        numWorkers: int,
        shuffle: bool,
        idxs: List[int],
    ) -> Iterator[Tuple[Tensor, List[str], Tensor]]:
        """
        @todo: fix the typing override.
        """
        embed = embedding()
        idx = 0
        idxs = deque(sorted(idxs))  # type: ignore
        done = False
        for p in range(4):
            path = Path(DocIterInit.base, f"partition-{p:08d}.parquet")
            file = pq.ParquetFile(path)
            N, D = file.metadata.num_rows, embedding.size
            batches = file.iter_batches(1, columns=["text"])
            for i, part in enumerate(batches):
                if idx == idxs[0]:
                    idxs.popleft()  # type: ignore
                    txt = part.column("text").to_pylist()
                    """
                    @todo: add forward_prefix to the Embedding interface.
                    """
                    vec, tokens, token_ids = embed.forward_prefix(txt)  # type: ignore
                    yield vec, tokens, token_ids.detach().cpu().tolist()
                idx += 1
                if len(idxs) == 0:
                    done = True
                    break
            if done:
                break

    def getDocLen(self) -> int:
        docLen = 0
        for i in range(DocIterInit.N):
            file = pq.ParquetFile(Path(DocIterInit.base, f"partition-{i:08d}.parquet"))
            docLen += file.metadata.num_rows
        return docLen

    def qidIter(self, split: PartitionName, batchSize: int) -> Iterator[List[int]]:
        path = Path(QryIterInit.base, f"{split}.parquet")
        for part in pq.ParquetFile(path).iter_batches(batchSize, columns=["id"]):
            yield part.column("id").to_pylist()

    def qryIter(self, split: PartitionName, batchSize: int) -> Iterator[List[str]]:
        path = Path(QryIterInit.base, f"{split}.parquet")
        for part in pq.ParquetFile(path).iter_batches(batchSize, columns=["text"]):
            yield part.column("text").to_pylist()

    def qryEmbIter(
        self,
        embedding: Type[Embedding],
        split: PartitionName,
        batchSize: int,
        numWorkers: int,
        shuffle: bool,
    ) -> Iterator[Tensor]:
        path = Path(QryIterInit.base, f"{split}.parquet")
        file = pq.ParquetFile(path)
        path = Path(QryEmbIterInit.base, embedding.name, f"{split}.bin")
        N, D = file.metadata.num_rows, embedding.size
        storage = FloatStorage.from_file(str(path), False, N * D)
        samples = FloatTensor(storage).reshape(N, D)
        dataloader = DataLoader(
            TensorDataset(samples),
            batchSize,
            shuffle=shuffle,
            num_workers=numWorkers,
        )
        for batch in dataloader:
            yield batch[0]
        del samples

    def getQryLen(self, split: PartitionName) -> int:
        file = pq.ParquetFile(Path(QryIterInit.base, f"{split}.parquet"))
        return file.metadata.num_rows

    def getQryRel(self, split: PartitionName) -> Path:
        return Path(QryRelInit.base, f"{split}.tsv")

    def mixEmbIter(
        self,
        embedding: Type[Embedding],
        split: PartitionName,
        relevant: int,
        batchSize: int,
        numWorkers: int,
        shuffle: bool,
    ) -> Iterator[Tuple[Tensor, Tensor]]:
        if relevant > 256:
            raise ValueError("At most 256 documents per query.")
        # get the document embeddings
        buffer: List[Tensor] = []
        for i in range(DocIterInit.N):
            path = Path(DocIterInit.base, f"partition-{i:08d}.parquet")
            file = pq.ParquetFile(path)
            N, D = file.metadata.num_rows, embedding.size
            path = Path(DocEmbIterInit.base, embedding.name, f"partition-{i:08d}.bin")
            storage = FloatStorage.from_file(str(path), False, N * D)
            samples = FloatTensor(storage).reshape(N, D)
            buffer.append(samples)
        docEmb = torch.cat(buffer, dim=0)
        # get the query embeddings
        path = Path(QryIterInit.base, f"{split}.parquet")
        file = pq.ParquetFile(path)
        path = Path(QryEmbIterInit.base, embedding.name, f"{split}.bin")
        N, D = file.metadata.num_rows, embedding.size
        storage = FloatStorage.from_file(str(path), False, N * D)
        qryEmb = FloatTensor(storage).reshape(N, D)
        # get the document indices associated with each query
        path = Path(MixEmbIterInit.base, embedding.name, f"{split}.bin")
        N, K = file.metadata.num_rows, 256
        storage = IntStorage.from_file(str(path), False, N * K)
        indices = IntTensor(storage).reshape(N, K)

        # one-to-many mapping from query to documents
        class Dataset(torch.utils.data.Dataset):

            def __len__(self) -> int:
                return N

            def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
                return qryEmb[index], docEmb[indices[index][:relevant]]

        dataloader = DataLoader(
            Dataset(),
            batchSize,
            shuffle=shuffle,
            num_workers=numWorkers,
        )
        for batch in dataloader:
            yield tuple(batch)
        del docEmb, qryEmb, indices

    def getMixLen(self, split: PartitionName) -> int:
        return self.getQryLen(split)


class DocIterInit:
    """
    Initialize the document iterator for MsMarco.
    """

    N = 4
    base = Path(workspace, f"{MsMarcoDataset.name}/doc")

    def __init__(self) -> None:
        console.log("Prepare DocIter for MsMarco")
        self.base.mkdir(mode=0o770, parents=True, exist_ok=True)
        asyncio.run(self.dispatch())

    async def dispatch(self) -> None:
        await download(
            "https://msmarco.z22.web.core.windows.net/msmarcoranking/collection.tar.gz",
            Path(self.base, "collection.tar.gz"),
        )
        subprocess.run(
            ["tar", "-xzvf", "collection.tar.gz"],
            cwd=self.base,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        Path(self.base, "collection.tar.gz").unlink()
        Path(self.base, "collection.tsv").chmod(0o770)
        ids: List[List[int]] = [[] for _ in range(self.N)]
        texts: List[List[str]] = [[] for _ in range(self.N)]
        with open(Path(self.base, "collection.tsv"), "r") as f:
            for line in f:
                parts = [p.strip() for p in line.split("\t")]
                x0, x1 = int(parts[0]), parts[1]
                ids[x0 % self.N].append(x0)
                texts[x0 % self.N].append(x1)
        for i in range(self.N):
            table = pa.Table.from_pydict({"id": ids[i], "text": texts[i]})
            pq.write_table(table, Path(self.base, f"partition-{i:08d}.parquet"))
        Path(self.base, "collection.tsv").unlink()


class DocEmbIterInit:
    """
    Initialize the document embedding iterator for MsMarco.
    """

    base = Path(workspace, f"{MsMarcoDataset.name}/docEmb")

    def __init__(self, embedding: Embedding, partition: int) -> None:
        console.log("Prepare DocEmbIter for MsMarco, Partition", partition)
        self.base = Path(self.base, embedding.name)
        self.base.mkdir(mode=0o770, parents=True, exist_ok=True)
        self.embedding, self.partition = embedding, partition
        self.dispatch()

    def dispatch(self, batchSize: int = 128) -> None:
        path = Path(DocIterInit.base, f"partition-{self.partition:08d}.parquet")
        file = pq.ParquetFile(path)
        path = Path(self.base, f"partition-{self.partition:08d}.bin")
        N, K = file.metadata.num_rows, self.embedding.size
        storage = FloatStorage.from_file(str(path), True, N * K)
        samples = FloatTensor(storage).reshape(N, K)
        with Progress(console=console) as progress:
            T = progress.add_task("Embedding", total=N)
            batches = file.iter_batches(batchSize, columns=["text"])
            for i, part in enumerate(batches):
                txt = part.column("text").to_pylist()
                vec = self.embedding.forward(txt)
                samples[i * batchSize : (i + 1) * batchSize] = vec
                progress.advance(T, batchSize)
                torch.cuda.empty_cache()
            progress.remove_task(T)


class QryIterInit:
    """
    Initialize the query iterator for MsMarco.
    """

    base = Path(workspace, f"{MsMarcoDataset.name}/qry")

    def __init__(self) -> None:
        console.log("Prepare QryIter for MsMarco")
        self.base.mkdir(mode=0o770, parents=True, exist_ok=True)
        asyncio.run(self.dispatch())

    async def dispatch(self) -> None:
        await download(
            "https://msmarco.z22.web.core.windows.net/msmarcoranking/queries.tar.gz",
            Path(self.base, "queries.tar.gz"),
        )
        subprocess.run(
            ["tar", "-xzvf", "queries.tar.gz"],
            cwd=self.base,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        Path(self.base, "queries.tar.gz").unlink()
        Path(self.base, "queries.eval.tsv").unlink()
        ids: List[int] = []
        texts: List[str] = []
        with open(Path(self.base, "queries.train.tsv"), "r") as f:
            for line in f:
                parts = [p.strip() for p in line.split("\t")]
                x0, x1 = int(parts[0]), parts[1]
                ids.append(x0)
                texts.append(x1)
        table = pa.Table.from_pydict({"id": ids, "text": texts})
        pq.write_table(table, Path(self.base, "Train.parquet"))
        ids.clear()
        texts.clear()
        with open(Path(self.base, "queries.dev.tsv"), "r") as f:
            for line in f:
                parts = [p.strip() for p in line.split("\t")]
                x0, x1 = int(parts[0]), parts[1]
                ids.append(x0)
                texts.append(x1)
        table = pa.Table.from_pydict({"id": ids, "text": texts})
        pq.write_table(table, Path(self.base, "Validate.parquet"))
        Path(self.base, "queries.train.tsv").unlink()
        Path(self.base, "queries.dev.tsv").unlink()


class QryEmbIterInit:
    """
    Initialize the query embedding iterator for MsMarco.
    """

    base = Path(workspace, f"{MsMarcoDataset.name}/qryEmb")

    def __init__(self, embedding: Embedding) -> None:
        console.log("Prepare QryEmbIter for MsMarco")
        self.base = Path(self.base, embedding.name)
        self.base.mkdir(mode=0o770, parents=True, exist_ok=True)
        self.embedding = embedding
        self.dispatch()

    def dispatch(self, batchSize: int = 256) -> None:
        for item in ["Train", "Validate"]:
            path = Path(QryIterInit.base, f"{item}.parquet")
            file = pq.ParquetFile(path)
            path = Path(self.base, f"{item}.bin")
            N, K = file.metadata.num_rows, self.embedding.size
            storage = FloatStorage.from_file(str(path), True, N * K)
            samples = FloatTensor(storage).reshape(N, K)
            with Progress(console=console) as progress:
                T = progress.add_task("Embedding", total=N)
                batches = file.iter_batches(batchSize, columns=["text"])
                for i, part in enumerate(batches):
                    txt = part.column("text").to_pylist()
                    vec = self.embedding.forward(txt)
                    samples[i * batchSize : (i + 1) * batchSize] = vec
                    progress.advance(T, batchSize)
                    torch.cuda.empty_cache()
                progress.remove_task(T)


class QryRelInit:
    """
    Initialize the query relevance for MsMarco.
    """

    base = Path(workspace, f"{MsMarcoDataset.name}/qryRel")

    def __init__(self) -> None:
        console.log("Prepare QryRel for MsMarco")
        self.base.mkdir(mode=0o770, parents=True, exist_ok=True)
        asyncio.run(self.dispatch())

    async def dispatch(self):
        # we should have dispatched all tasks at once, but due to progress bar
        # constraints, only one at a time is possible. Otherwise, the progress
        # bar would be globally defined, and may interfere with training logs.
        await download(
            "https://msmarco.z22.web.core.windows.net/msmarcoranking/qrels.dev.tsv",
            Path(self.base, "Validate.tsv"),
        )
        await download(
            "https://msmarco.z22.web.core.windows.net/msmarcoranking/qrels.train.tsv",
            Path(self.base, "Train.tsv"),
        ),


class MixEmbIterInit:
    """
    Initialize the mixed embedding iterator for MsMarco.
    """

    base = Path(workspace, f"{MsMarcoDataset.name}/mixEmb")

    def __init__(self, embedding: Type[Embedding]) -> None:
        self.base = Path(self.base, embedding.name)
        self.base.mkdir(mode=0o770, parents=True, exist_ok=True)
        index = faiss.IndexFlatIP(embedding.size)
        devices = [i for i in range(cuda.device_count())]
        resources = [faiss.StandardGpuResources() for _ in range(len(devices))]
        clonerOptions = faiss.GpuMultipleClonerOptions()
        clonerOptions.shard = True
        args = (resources, index, clonerOptions, devices)
        self.index = faiss.index_cpu_to_gpu_multiple_py(*args)
        self.dispatch(embedding)

    def dispatch(
        self,
        embedding: Type[Embedding],
        readSize: int = 8192,
        saveSize: int = 512,
    ) -> None:
        dataset = MsMarcoDataset()
        docEmbs = np.empty((dataset.getDocLen(), embedding.size), dtype=np.float32)
        with Progress(console=console) as progress:
            T = progress.add_task("Indexing", total=dataset.getDocLen())
            for i, batch in enumerate(
                dataset.docEmbIter(embedding, readSize, 8, False)
            ):
                docEmbs[i * readSize : (i + 1) * readSize] = batch.numpy()
                progress.advance(T, readSize)
            self.index.add(docEmbs)
            items: List[PartitionName] = ["Train", "Validate"]
            for item in items:
                path = Path(self.base, f"{item}.bin")
                N, K = dataset.getQryLen(item), 256
                storage = IntStorage.from_file(str(path), True, N * K)
                samples = IntTensor(storage).reshape(N, K)
                T = progress.add_task("Querying", total=N)
                for i, batch in enumerate(
                    dataset.qryEmbIter(embedding, item, saveSize, 8, False)
                ):
                    _, I = self.index.search(batch.numpy(), 256)
                    I = torch.from_numpy(I).to(torch.int32)
                    samples[i * saveSize : (i + 1) * saveSize] = I
                    progress.advance(T, saveSize)


def main():
    """
    Initialize the MsMarco dataset.
    """
    DocIterInit()
    DocEmbIterInit(BgeBaseEmbedding(), partition=0)
    DocEmbIterInit(BgeBaseEmbedding(), partition=1)
    DocEmbIterInit(BgeBaseEmbedding(), partition=2)
    DocEmbIterInit(BgeBaseEmbedding(), partition=3)
    DocEmbIterInit(MiniPcmEmbedding(), partition=0)
    DocEmbIterInit(MiniPcmEmbedding(), partition=1)
    DocEmbIterInit(MiniPcmEmbedding(), partition=2)
    DocEmbIterInit(MiniPcmEmbedding(), partition=3)
    QryIterInit()
    QryEmbIterInit(BgeBaseEmbedding())
    QryEmbIterInit(MiniPcmEmbedding())
    MixEmbIterInit(BgeBaseEmbedding)
    MixEmbIterInit(MiniPcmEmbedding)
    QryRelInit()


if __name__ == "__main__":
    main()
