import shutil
import argparse
import subprocess
from typing import Type
from pathlib import Path
import numpy as np
from importlib import import_module
from elasticsearch import Elasticsearch, helpers
from rich.progress import Progress
from source import console
from source.interpret.retrieval import workspace
from source.model import workspace as modelWorkspace
from source.utilities.model import saveComputed
from source.dataset import MsMarcoDataset, BeirDataset
from source.embedding import BgeBaseEmbedding
from source.interface import Dataset, Embedding


def main(
    embedding: Type[Embedding], dataset: Dataset, version: str, esHost: str, esPort: int
):
    # load attributes for latent space
    module = import_module(f"source.model.{version}")
    activate = module.Trainer.hyperParams.activate
    assert isinstance(activate, int)

    # define where to read computed features
    readBase = Path(modelWorkspace, version, "computed", dataset.name)
    if not readBase.exists():
        saveComputed(embedding, dataset, version)
    docLen = dataset.getDocLen()
    docLatentIndex = np.memmap(
        Path(readBase, "docLatentIndex.bin"),
        dtype=np.int32,
        shape=(docLen, activate),
    )
    docLatentValue = np.memmap(
        Path(readBase, "docLatentValue.bin"),
        dtype=np.float32,
        shape=(docLen, activate),
    )
    qryLen = dataset.getQryLen("Validate")
    qryLatentIndex = np.memmap(
        Path(readBase, "qryLatentIndex.bin"),
        dtype=np.int32,
        shape=(qryLen, activate),
    )
    qryLatentValue = np.memmap(
        Path(readBase, "qryLatentValue.bin"),
        dtype=np.float32,
        shape=(qryLen, activate),
    )

    # define where to save results
    saveBase = Path(workspace, "latent", dataset.name)
    saveBase.mkdir(mode=0o770, parents=True, exist_ok=True)
    qresFile = Path(saveBase, f"{version}.qres")
    evalFile = Path(saveBase, f"{version}.eval")

    # create connection to elastic search
    es = Elasticsearch(
        hosts=[{"host": esHost, "port": esPort, "scheme": "http"}],
    )
    es.indices.create(
        index=f"{version}.latent".lower(),
        body={"mappings": {"properties": {"vector": {"type": "sparse_vector"}}}},
        ignore=400,
    )

    with Progress(console=console) as p:

        # build index with elastic search
        t, a = p.add_task("Indexing...", total=docLen), []
        dids = [d for batch in dataset.didIter(4096) for d in batch]
        for i in range(0, docLen):
            vector = {}
            for j in range(0, activate):
                key = docLatentIndex[i, j]
                val = docLatentValue[i, j]
                if val <= 0.0:
                    continue
                vector[str(key)] = val
            a.append(
                {
                    "_index": f"{version}.latent".lower(),
                    "_id": dids[i],
                    "_source": {"vector": vector},
                }
            )
            # bulk insert
            if len(a) == 4096 or i == docLen - 1:
                helpers.bulk(es, a)
                p.update(t, advance=len(a))
                a.clear()
        p.stop_task(t)

        # batch query with elastic search
        t, a, b = p.add_task("Querying...", total=qryLen), [], []
        qids = [q for batch in dataset.qidIter("Validate", 4096) for q in batch]
        with qresFile.open("w") as f:
            for i in range(0, qryLen):
                vector = {}
                for j in range(0, activate):
                    key = qryLatentIndex[i, j]
                    val = qryLatentValue[i, j]
                    if val <= 0.0:
                        continue
                    vector[str(key)] = val
                a.append(
                    {
                        "index": f"{version}.latent".lower(),
                    }
                )
                a.append(
                    {
                        "query": {
                            "sparse_vector": {
                                "field": "vector",
                                "query_vector": vector,
                            }
                        },
                        "size": 100,
                    }
                )
                b.append(qids[i])
                # bulk search
                if len(a) == 32 or i == qryLen - 1:
                    rsp = es.msearch(
                        body=a,
                        request_timeout=180,
                        max_concurrent_searches=32,
                    )
                    for j, res in enumerate(rsp["responses"]):
                        qid = b[j]
                        for hit in res["hits"]["hits"]:
                            f.write(
                                f"{qid}\tQ0\t{hit['_id']}\t0\t{hit['_score']}\tLatent\n"
                            )
                    f.flush()
                    p.advance(t, advance=len(b))
                    a, b = [], []
        p.stop_task(t)

    # evaluate results with trec_eval
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
    parser.add_argument("version", type=str)
    parser.add_argument("--esHost", type=str, default="localhost")
    parser.add_argument("--esPort", type=int, default=9200)
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
    main(embedding, dataset, args.version, args.esHost, args.esPort)
