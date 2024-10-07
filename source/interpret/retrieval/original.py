import faiss
import argparse
import subprocess
from typing import Type
from pathlib import Path
from rich.progress import Progress
from source import console
from source.interpret.retrieval import workspace
from source.dataset import MsMarcoDataset, BeirDataset
from source.embedding import BgeBaseEmbedding
from source.interface import Dataset, Embedding


def main(embedding: Type[Embedding], dataset: Dataset) -> None:
    # define where to save results
    saveBase = Path(workspace, "original")
    saveBase.mkdir(mode=0o770, parents=True, exist_ok=True)
    qresFile = Path(saveBase, f"{embedding.name}{dataset.name}.qres")
    evalFile = Path(saveBase, f"{embedding.name}{dataset.name}.eval")

    # build index with faiss across GPUs
    cpuIndex = faiss.IndexFlatIP(embedding.size)
    with Progress(console=console) as p:
        t = p.add_task("Indexing...", total=dataset.getDocLen())
        iterator = dataset.docEmbIter(embedding, 4096, 8, False)
        for docEmb in iterator:
            cpuIndex.add(docEmb.numpy())
            p.update(t, advance=docEmb.size(0))
        p.remove_task(t)
    gpuIndex = faiss.index_cpu_to_all_gpus(cpuIndex)

    # perform retrieval on the validation set
    dids = [d for batch in dataset.didIter(4096) for d in batch]
    with Progress(console=console) as p:
        t = p.add_task("Retrieving...", total=dataset.getQryLen("Validate"))
        qryIterator = dataset.qryEmbIter(embedding, "Validate", 512, 4, False)
        qidIterator = dataset.qidIter("Validate", 512)
        with qresFile.open("w") as f:
            for qrys, qids in zip(qryIterator, qidIterator):
                D, I = gpuIndex.search(qrys.numpy(), 100)
                for qid, sims, dnos in zip(qids, D, I):
                    for s, d in zip(sims, dnos):
                        f.write(f"{qid}\tQ0\t{dids[d]}\t0\t{s}\tOriginal\n")
                f.flush()
                p.update(t, advance=qrys.size(0))
        p.remove_task(t)

    # evaluate the retrieval results
    args = []
    args.append("trec_eval")
    args.extend(["-m", "all_trec"])
    args.append(dataset.getQryRel("Validate").as_posix())
    args.append(qresFile.as_posix())
    with console.status("Evaluating..."):
        with evalFile.open("w") as f:
            subprocess.run(args, stdout=f)


if __name__ == "__main__":
    # specify command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("embedding", type=str, choices=["BgeBase"])
    parser.add_argument("dataset", type=str, choices=["MsMarco", "Beir"])
    args = parser.parse_args()

    # parse arguments into concrete instances
    match args.dataset:
        case "MsMarco":
            dataset = MsMarcoDataset()
        case "Beir":
            dataset = BeirDataset()
        case _:
            raise NotImplementedError()
    match args.embedding:
        case "BgeBase":
            embedding = BgeBaseEmbedding
        case _:
            raise NotImplementedError()

    # run the workflow
    main(embedding, dataset)
