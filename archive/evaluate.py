import io
import json
import torch
import argparse
import importlib
import bm25s
from bm25s.tokenization import Tokenized
import subprocess
from typing import List
from pathlib import Path
from source import progress
from source.model import workspace
from source.dataset.msMarco import MsMarcoDataset
from source.embedding.bgeBase import BgeBaseEmbedding
from source.interface import SAE, SubsetSplit


def convert_to_trec(results, scores, query_ids, doc_ids, run_id="bm25_run"):
    lines = []
    num_queries, num_results = results.shape

    for query_idx in range(num_queries):
        query_id = query_ids[query_idx]  # Get the query ID for the current query
        for rank in range(num_results):
            doc_index = results[query_idx, rank]  # Index in the doc_ids array
            doc = doc_ids[doc_index]  # Retrieve the actual document ID
            score = scores[query_idx, rank]  # Score for the document
            # Rank starts from 1 in TREC format
            trec_line = f"{query_id} Q0 {doc} {rank + 1} {score:.2f} {run_id}"
            lines.append(trec_line)
    return lines


class MsMarcoBgeBase:

    N: int = 4

    def __init__(self) -> None:
        self.dataset = MsMarcoDataset()

    @torch.inference_mode()
    def reconFAISS(self, version: str, batchSize: int = 4096):

        # define workspace
        base = Path(workspace, version)
        assert base.exists() and base.is_dir()

    @torch.inference_mode()
    def latentBM25(self, version: str, batchSize: int = 4096):

        # define workspace
        base = Path(workspace, version)
        print(base)
        assert base.exists() and base.is_dir()
        docsDir = Path(base, "latentBM25/docs")
        docsDir.mkdir(mode=0o770, parents=True, exist_ok=True)
        indexDir = Path(base, "latentBM25/index")
        indexDir.mkdir(mode=0o770, parents=True, exist_ok=True)
        qrysFile = Path(base, "latentBM25/qrys.tsv")
        qresFile = Path(base, "latentBM25/qres.tsv")

        # # load model into memory
        module = importlib.import_module(f"source.model.archive.{version}")
        Model, Trainer = module.Model, module.Trainer
        state = max(Path(base, "state").glob("*.pth"))
        model = Model(BgeBaseEmbedding.size, Trainer.hyperParams.expandBy)
        assert isinstance(model, SAE) and isinstance(model, torch.nn.Module)
        model.load_state_dict(torch.load(state, map_location="cpu")["model"])
        model.eval().cuda()

        # # collect document features
        dids = self.dataset.didIter(batchSize)
        # docs = self.dataset.docEmbIter(BgeBaseEmbedding, batchSize, 4, False)
        # # fds: List[io.TextIOWrapper] = []
        # # for i in range(self.N):
        # #     fds.append(Path(docsDir, f"partition-{i:08d}.jsonl").open("w"))
        # T = progress.add_task("Embedding", total=self.dataset.getDocLen())
        # documents = []
        # for batchX, batchY in zip(dids, docs):
        #     _, batchF = model.forward(batchY.cuda(), Trainer.hyperParams.activate)
        #     progress.advance(T, batchF.size(0))
        #     for x, f in zip(batchX, batchF):
        #         doc = " ".join(map(str, f.nonzero().squeeze().tolist()))
        #         documents.append(doc)

        #         # fds[x % self.N].write(json.dumps(line) + "\n")
        # # for fd in fds:
        # #     fd.close()
        # # vocab = {k: v for k,v in zip(map(str, vocab), vocab)}
        # corpus_tokens = bm25s.tokenize(documents, stopwords="en")
        # progress.remove_task(T)

        # build BM25 index on latent features
        # retriever = bm25s.BM25()
        # retriever.index(corpus_tokens)
        retriever = bm25s.BM25.load(
            f"/data/user_data/tevinw/esae/{version}-bm25s", load_corpus=False, mmap=True
        )
        # args = []
        # args.extend(["conda", "run", "--live-stream", "-n", "esae"])
        # args.extend(["python3", "-m", "pyserini.index.lucene"])
        # args.extend(["--collection", "JsonCollection"])
        # args.extend(["--input", docsDir.as_posix(), "--index", indexDir.as_posix()])
        # args.extend(["--generator", "DefaultLuceneDocumentGenerator"])
        # subprocess.run(args, check=True)

        # collect query features
        split: SubsetSplit = "Validate"
        qids = self.dataset.qidIter("Validate", batchSize)
        qrys = self.dataset.qryEmbIter(BgeBaseEmbedding, split, batchSize, 4, False)
        T = progress.add_task("Embedding", total=self.dataset.getQryLen(split))
        # all_query_ids = []
        # vocab = set()
        queries = []
        with qrysFile.open("w") as fd:
            for batchX, batchY in zip(qids, qrys):
                _, batchF = model.forward(batchY.cuda(), Trainer.hyperParams.activate)
                progress.advance(T, batchF.size(0))
                for x, f in zip(batchX, batchF):
                    query = " ".join(map(str, f.nonzero().squeeze().tolist()))
                    queries.append(query)
        progress.remove_task(T)

        query_tokens = bm25s.tokenize(queries, stopwords="en")

        # search BM25 index
        results, scores = retriever.retrieve(query_tokens, k=2, n_threads=30)
        trec_lines = convert_to_trec(
            results,
            scores,
            [qid for qid_batch in qids for qid in qid_batch],
            [did for did_batch in dids for did in did_batch],
        )
        with open(f"{version}-results.trec", "w") as f:
            for line in trec_lines:
                f.write(line + "\n")

        # args = []
        # args.extend(["conda", "run", "--live-stream", "-n", "esae"])
        # args.extend(["python3", "-m", "pyserini.search.lucene"])
        # args.extend(["--index", indexDir.as_posix()])
        # args.extend(["--topics", qrysFile.as_posix()])
        # args.extend(["--output", qresFile.as_posix()])
        # args.extend(["--threads", "32"])
        # args.extend(["--bm25"])
        # args.extend(["--hits", "100"])
        # args.extend(["--batch-size", "512"])
        # subprocess.run(args, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("version", type=str)
    E = MsMarcoBgeBase()
    E.latentBM25(parser.parse_args().version)