"""
@brief: Implementation for BEIR dataset.
@author: Hao Kang <haok@andrew.cmu.edu>
@date: October 06, 2024
"""

import faiss
import torch
import asyncio
import numpy as np
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
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from source import console
from source.interface import Dataset, Embedding, PartitionName
from source.dataset import workspace
from source.embedding.bgeBase import BgeBaseEmbedding


class BeirDataset(Dataset):

    name = "Beir"

    def didIter(self, batchSize: int) -> Iterator[List[str]]:
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

    def qidIter(self, split: PartitionName, batchSize: int) -> Iterator[List[str]]:
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
        raise NotImplementedError("Query relevance file not available.")
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
    Initialize the document iterator for Beir.
    """

    N = 4
    base = Path(workspace, f"{BeirDataset.name}/doc")

    def __init__(self) -> None:
        console.log("Prepare DocIter for Beir")
        self.base.mkdir(mode=0o770, parents=True, exist_ok=True)
        self.dispatch()

    def dispatch(self) -> None:
        ids: List[str] = [[] for _ in range(self.N)]
        texts: List[str] = [[] for _ in range(self.N)]
        url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip"  # fmt: off
        for dataset in ["trec-covid", "nq", "dbpedia-entity"]:
            folder = util.download_and_unzip(url.format(dataset), "data")
            loader = GenericDataLoader(folder)
            corpus, _, _ = loader.load(split="test")
            for key, val in corpus.items():
                assert isinstance(key, str)
                assert isinstance(val, dict) and "text" in val
                assert isinstance(val["text"], str)
                ids[hash(key) % self.N].append(key)
                texts[hash(key) % self.N].append(val["text"])
        for i in range(self.N):
            table = pa.Table.from_pydict({"id": ids[i], "text": texts[i]})
            pq.write_table(table, Path(self.base, f"partition-{i:08d}.parquet"))


class DocEmbIterInit:
    """
    Initialize the document embedding iterator for Beir.
    """

    base = Path(workspace, f"{BeirDataset.name}/docEmb")

    def __init__(self, embedding: Embedding, partition: int) -> None:
        console.log("Prepare DocEmbIter for Beir, Partition", partition)
        self.base = Path(self.base, embedding.name)
        self.base.mkdir(mode=0o770, parents=True, exist_ok=True)
        self.embedding, self.partition = embedding, partition
        self.dispatch()

    def dispatch(self, batchSize: int = 1024) -> None:
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
            progress.remove_task(T)


class QryIterInit:
    """
    Initialize the query iterator for Beir.
    """

    base = Path(workspace, f"{BeirDataset.name}/qry")

    def __init__(self) -> None:
        console.log("Prepare QryIter for Beir")
        self.base.mkdir(mode=0o770, parents=True, exist_ok=True)
        asyncio.run(self.dispatch())

    async def dispatch(self) -> None:
        ids: List[int] = []
        texts: List[str] = []
        url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip"  # fmt: off
        for dataset in ["trec-covid", "nq", "dbpedia-entity"]:
            folder = util.download_and_unzip(url.format(dataset), "data")
            loader = GenericDataLoader(folder)
            _, queries, _ = loader.load(split="test")
            for key, val in queries.items():
                assert isinstance(key, str)
                assert isinstance(val, str)
                ids.append(key)
                texts.append(val)
        table = pa.Table.from_pydict({"id": ids, "text": texts})
        pq.write_table(table, Path(self.base, "Validate.parquet"))


class QryEmbIterInit:
    """
    Initialize the query embedding iterator for Beir.
    """

    base = Path(workspace, f"{BeirDataset.name}/qryEmb")

    def __init__(self, embedding: Embedding) -> None:
        console.log("Prepare QryEmbIter for Beir")
        self.base = Path(self.base, embedding.name)
        self.base.mkdir(mode=0o770, parents=True, exist_ok=True)
        self.embedding = embedding
        self.dispatch()

    def dispatch(self, batchSize: int = 1024) -> None:
        for item in ["Validate"]:
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
                progress.remove_task(T)


class QryRelInit:
    """
    Initialize the query relevance for Beir.
    """

    base = Path(workspace, f"{BeirDataset.name}/qryRel")

    def __init__(self) -> None:
        console.log("Prepare QryRel for Beir")
        self.base.mkdir(mode=0o770, parents=True, exist_ok=True)
        self.dispatch()

    def dispatch(self) -> None:
        url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip"  # fmt: off
        with Path(self.base, "Validate.tsv").open("w") as f:
            for dataset in ["trec-covid", "nq", "dbpedia-entity"]:
                folder = util.download_and_unzip(url.format(dataset), "data")
                _, _, qrels = GenericDataLoader(folder).load(split="test")
                for qid, val in qrels.items():
                    assert isinstance(qid, str)
                    for did, rel in val.items():
                        assert isinstance(did, str)
                        assert isinstance(rel, int)
                        f.write(f"{qid}\t0\t{did}\t{rel}\n")
                f.flush()


class MixEmbIterInit:
    """
    Initialize the mixed embedding iterator for Beir.
    """

    base = Path(workspace, f"{BeirDataset.name}/mixEmb")

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
        dataset = BeirDataset()
        docEmbs = np.empty((dataset.getDocLen(), embedding.size), dtype=np.float32)
        with Progress(console=console) as progress:
            T = progress.add_task("Indexing", total=dataset.getDocLen())
            for i, batch in enumerate(
                dataset.docEmbIter(embedding, readSize, 8, False)
            ):
                docEmbs[i * readSize : (i + 1) * readSize] = batch.numpy()
                progress.advance(T, readSize)
            self.index.add(docEmbs)
            path = Path(self.base, f"Validate.bin")
            N, K = dataset.getQryLen("Validate"), 256
            storage = IntStorage.from_file(str(path), True, N * K)
            samples = IntTensor(storage).reshape(N, K)
            T = progress.add_task("Querying", total=N)
            for i, batch in enumerate(
                dataset.qryEmbIter(embedding, "Validate", saveSize, 8, False)
            ):
                _, I = self.index.search(batch.numpy(), 256)
                I = torch.from_numpy(I).to(torch.int32)
                samples[i * saveSize : (i + 1) * saveSize] = I
                progress.advance(T, saveSize)


def main():
    """
    Initialize the MsMarco dataset.
    """
    # DocIterInit()
    # DocEmbIterInit(BgeBaseEmbedding(), partition=0)
    # DocEmbIterInit(BgeBaseEmbedding(), partition=1)
    # DocEmbIterInit(BgeBaseEmbedding(), partition=2)
    # DocEmbIterInit(BgeBaseEmbedding(), partition=3)
    # QryIterInit()
    # QryEmbIterInit(BgeBaseEmbedding())
    # MixEmbIterInit(BgeBaseEmbedding)
    QryRelInit()


if __name__ == "__main__":
    main()
