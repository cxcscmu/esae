import faiss
import shutil
import argparse
import subprocess
from typing import Type
from pathlib import Path
import numpy as np
from rich.progress import Progress
from source import console
from source.interpret.retrieval import workspace
from source.model import workspace as modelWorkspace
from source.utilities.model import saveComputed
from source.dataset import MsMarcoDataset
from source.embedding import BgeBaseEmbedding
from source.interface import Dataset, Embedding


def main(dataset: Dataset, embedding: Type[Embedding], version: str):

    # define where to read computed features
    readBase = Path(modelWorkspace, version, "computed")
    if not readBase.exists():
        saveComputed(dataset, embedding, version)
    docDecode = np.memmap(
        Path(readBase, "docDecode.bin"),
        dtype=np.float32,
        shape=(dataset.getDocLen(), embedding.size),
    )
    qryDecode = np.memmap(
        Path(readBase, "qryDecode.bin"),
        dtype=np.float32,
        shape=(dataset.getQryLen("Validate"), embedding.size),
    )

    # define where to save results
    saveBase = Path(workspace, "decoded")
    saveBase.mkdir(mode=0o770, parents=True, exist_ok=True)
    qresFile = Path(saveBase, f"{version}.qres")
    evalFile = Path(saveBase, f"{version}.eval")

    # build index with faiss across GPUs
    cpuIndex = faiss.IndexFlatIP(embedding.size)
    cpuIndex.add(docDecode)
    gpuIndex = faiss.index_cpu_to_all_gpus(cpuIndex)

    # perform retrieval for validation set
    dids = [d for batch in dataset.didIter(4096) for d in batch]
    qids = [q for batch in dataset.qidIter("Validate", 512) for q in batch]
    with Progress(console=console) as p:
        t = p.add_task("Retrieving...", total=qryDecode.shape[0])
        with qresFile.open("w") as f:
            for i in range(0, qryDecode.shape[0], 512):
                bQrys, bQids = qryDecode[i : i + 512], qids[i : i + 512]
                D, I = gpuIndex.search(bQrys, 100)
                for qid, sims, dnos in zip(bQids, D, I):
                    for s, d in zip(sims, dnos):
                        f.write(f"{qid}\tQ0\t{dids[d]}\t0\t{s}\tDecoded\n")
                f.flush()
                p.update(t, advance=bQrys.shape[0])
        p.remove_task(t)

    # evaluate results with trec_eval
    args = []
    args.append("trec_eval")
    args.extend(["-m", "all_trec"])
    args.append(dataset.getQryRel("Validate").as_posix())
    args.append(qresFile.as_posix())
    with console.status("Evaluating..."):
        with evalFile.open("w") as f:
            subprocess.run(args, stdout=f)
    shutil.copy(evalFile, "last.log")


if __name__ == "__main__":
    # specify command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("version", type=str)
    parser.add_argument("--dataset", type=str, default="MsMarco", choices=["MsMarco"])
    parser.add_argument("--embedding", type=str, default="BgeBase", choices=["BgeBase"])
    args = parser.parse_args()

    # parse arguments into concrete instances
    match args.dataset:
        case "MsMarco":
            dataset = MsMarcoDataset()
        case _:
            raise NotImplementedError()
    match args.embedding:
        case "BgeBase":
            embedding = BgeBaseEmbedding
        case _:
            raise NotImplementedError()

    # run the workflow
    main(dataset, embedding, args.version)
