import torch
import argparse
import importlib
import torch.nn as nn
import numpy as np
import bm25s
import faiss
from torch import Tensor
from rich.progress import Progress
from typing import List
from pathlib import Path
from source import console
from source.model import workspace
from source.embedding.bgeBase import BgeBaseEmbedding
from source.dataset.msMarco import MsMarcoDataset


@torch.inference_mode()
def saveModelOutput(version: str, batchSize: int = 8192):
    """
    Save the output of the model to disk.

    :param version: version of the model to evaluate
    :param batchSize: batch size for inference
    """

    # sanity check
    base = Path(workspace, version)
    assert base.exists() and base.is_dir()

    # load model into memory
    module = importlib.import_module(f"source.model.{version}")
    Model, Trainer = module.Model, module.Trainer
    snapshot = max(Path(base, "snapshot").glob("*.pth"))
    features = Trainer.hyperParams.features
    expandBy = Trainer.hyperParams.expandBy
    assert isinstance(features, int) and isinstance(expandBy, int)
    model = Model(features, expandBy)
    assert isinstance(model, nn.Module)
    model.load_state_dict(torch.load(snapshot, map_location="cpu")["model"])
    model.eval().cuda()

    # define where to save features
    save = Path(base, "computed")
    save.mkdir(mode=0o770, parents=True, exist_ok=True)

    dataset = MsMarcoDataset()
    activate = Trainer.hyperParams.activate
    assert isinstance(activate, int)
    with Progress(console=console) as progress:
        # create memory-mapped files to avoid RAM overflow
        featSave = np.memmap(
            Path(save, f"docLatent.npy"),
            dtype=np.int32,
            mode="w+",
            shape=(dataset.getDocLen(), activate),
        )
        xhatSave = np.memmap(
            Path(save, "docDecode.npy"),
            dtype=np.float32,
            mode="w+",
            shape=(dataset.getDocLen(), features),
        )

        # collect document features
        T = progress.add_task("DocEmb", total=dataset.getDocLen())
        for i, x in enumerate(
            dataset.docEmbIter(BgeBaseEmbedding, batchSize, 8, False)
        ):
            f, xhat = model.forward(x.cuda(), activate)
            assert isinstance(f, Tensor) and isinstance(xhat, Tensor)
            indices = torch.topk(f, activate, dim=1).indices
            indices = indices.detach().cpu().numpy()
            featSave[i * batchSize : (i + 1) * batchSize] = indices
            xhat = xhat.detach().cpu().numpy()
            xhatSave[i * batchSize : (i + 1) * batchSize] = xhat
            progress.advance(T, x.size(0))
        del featSave, xhatSave
        progress.stop_task(T)

        # create memory-mapped files to avoid RAM overflow
        featSave = np.memmap(
            Path(save, "qryLatent.npy"),
            dtype=np.int32,
            mode="w+",
            shape=(dataset.getQryLen("Validate"), activate),
        )
        xhatSave = np.memmap(
            Path(save, "qryDecode.npy"),
            dtype=np.float32,
            mode="w+",
            shape=(dataset.getQryLen("Validate"), features),
        )

        # collect query features
        T = progress.add_task("QryEmb", total=dataset.getQryLen("Validate"))
        for i, x in enumerate(
            dataset.qryEmbIter(BgeBaseEmbedding, "Validate", batchSize, 8, False)
        ):
            f, xhat = model.forward(x.cuda(), activate)
            assert isinstance(f, Tensor) and isinstance(xhat, Tensor)
            indices = torch.topk(f, activate, dim=1).indices
            indices = indices.detach().cpu().numpy()
            featSave[i * batchSize : (i + 1) * batchSize] = indices
            xhat = xhat.detach().cpu().numpy()
            xhatSave[i * batchSize : (i + 1) * batchSize] = xhat
            progress.advance(T, x.size(0))
        del featSave, xhatSave
        progress.stop_task(T)


def latentRetrieval(version: str):
    """
    Retrieve documents using latent features.

    :param version: version of the model to evaluate
    """

    # sanity check
    base = Path(workspace, version)
    assert base.exists() and base.is_dir()
    save = Path(base, "latentRetrieval")
    save.mkdir(mode=0o770, parents=True, exist_ok=True)

    Trainer = importlib.import_module(f"source.model.{version}").Trainer
    features = Trainer.hyperParams.features
    expandBy = Trainer.hyperParams.expandBy
    assert isinstance(features, int) and isinstance(expandBy, int)
    dataset = MsMarcoDataset()
    activate = Trainer.hyperParams.activate
    assert isinstance(activate, int)

    # initialize BM25 retriever
    retriever = bm25s.BM25()
    vocabulary = {str(i): i for i in range(features * expandBy)}

    # create memory-mapped files to avoid RAM overflow
    docF = np.memmap(
        Path(base, "computed", f"docLatent.npy"),
        dtype=np.int32,
        mode="r",
        shape=(dataset.getDocLen(), activate),
    )
    qryF = np.memmap(
        Path(base, "computed", f"qryLatent.npy"),
        dtype=np.int32,
        mode="r",
        shape=(dataset.getQryLen("Validate"), activate),
    )

    # build index with tokenized document features
    docTokenized = bm25s.tokenization.Tokenized(ids=docF, vocab=vocabulary)
    retriever.index(docTokenized)

    # collect queryIDs
    qids: List[int] = []
    for batch in dataset.qidIter("Validate", 4096):
        qids.extend(batch)
    dids: List[int] = []
    for batch in dataset.didIter(4096):
        dids.extend(batch)

    # retrieve top 100 documents for each query
    with Path(save, "results.tsv").open("w") as fd:
        for i in range(0, dataset.getQryLen("Validate"), 256):
            if i >= 4096:
                break
            batchQid, batchQry = qids[i : i + 256], qryF[i : i + 256]
            batchQry = bm25s.tokenization.Tokenized(ids=batchQry, vocab=vocabulary)
            results, scores = retriever.retrieve(batchQry, k=100, n_threads=24)
            for qid, result, score in zip(batchQid, results, scores):
                for off, sim in zip(result, score):
                    fd.write(f"{qid}\tQ0\t{dids[off]}\t0\t{sim}\t{version}\n")
                    fd.flush()


def decodeRetrieval(version: str):

    # sanity check
    base = Path(workspace, version)
    assert base.exists() and base.is_dir()
    save = Path(base, "decodeRetrieval")
    save.mkdir(mode=0o770, parents=True, exist_ok=True)
    Trainer = importlib.import_module(f"source.model.{version}").Trainer
    features = Trainer.hyperParams.features
    assert isinstance(features, int)

    # initialize the retriever
    dataset = MsMarcoDataset()
    index = faiss.IndexFlatL2(features)
    devices = [i for i in range(faiss.get_num_gpus())]
    resources = [faiss.StandardGpuResources() for _ in range(len(devices))]
    clonerOptions = faiss.GpuMultipleClonerOptions()
    clonerOptions.shard = True
    args = (resources, index, clonerOptions, devices)
    index = faiss.index_cpu_to_gpu_multiple_py(*args)

    # create memory-mapped files to avoid RAM overflow
    docXhat = np.memmap(
        Path(base, "computed", "docDecode.npy"),
        dtype=np.float32,
        mode="r",
        shape=(dataset.getDocLen(), features),
    )
    qryXhat = np.memmap(
        Path(base, "computed", "qryDecode.npy"),
        dtype=np.float32,
        mode="r",
        shape=(dataset.getQryLen("Validate"), features),
    )

    # collect queryIDs and documentIDs
    qids: List[int] = []
    for batch in dataset.qidIter("Validate", 4096):
        qids.extend(batch)
    dids: List[int] = []
    for batch in dataset.didIter(4096):
        dids.extend(batch)

    # add documents to the index
    index.add(docXhat)

    # retrieve top 100 documents for each query
    with Progress(console=console) as progress:
        T = progress.add_task("Retrieval", total=dataset.getQryLen("Validate"))
        with Path(save, f"results.tsv").open("w") as fd:
            for i in range(0, dataset.getQryLen("Validate"), 256):
                xhat = qryXhat[i : i + 256]
                D, I = index.search(xhat, 100)
                for qid, distances, indices in zip(qids[i : i + 256], D, I):
                    for dist, j in zip(distances, indices):
                        did, dist = dids[j], -1 * dist
                        fd.write(f"{qid}\tQ0\t{did}\t0\t{dist}\t{version}\n")
                progress.advance(T, 256)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("version", type=str, help="version of the model to evaluate")
    saveModelOutput(**vars(parser.parse_args()))
    latentRetrieval(**vars(parser.parse_args()))
    decodeRetrieval(**vars(parser.parse_args()))
